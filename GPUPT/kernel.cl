
typedef struct
{
    float4 orig;
	float4 dir;
} Ray;


float intersectSphere(const Ray *ray, float4 orig, float radius) // returns distance, 0 if no hit
{  
	float4 op = orig - ray->orig; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
	const float eps = 1e-4;
	float b = dot(op, ray->dir);
	float det = b * b - dot(op, op) + radius * radius;
   
	float ret = 0;

	if (det >= 0)
    {
		det = sqrt(det);

		float t = b - det;
		
		if (t > eps) ret = t;
		else
		{
			t = b + det;

			if (det > eps) ret =  t;
		}
	}

	return ret;
}

bool intersectSpheres(const Ray *ray, __global const float4 *position, __global const float *radius, int numSpheres, float *t, int *id)
{
    float inf = 1e20;
	*t = inf;
	
    for (int i = 0; i != numSpheres; ++i)
    {
        float d = intersectSphere(ray, position[i], radius[i]);

        if (d && d < *t) {
            *t = d;
            *id = i;
        }
    }
	
    return *t < inf;
}


const unsigned RAND48_SEED_0 = 0x330e;
const unsigned RAND48_SEED_1 = 0xabcd;
const unsigned RAND48_SEED_2 = 0x1234;
const unsigned RAND48_MULT_0 = 0xe66d;
const unsigned RAND48_MULT_1 = 0xdeec;
const unsigned RAND48_MULT_2 = 0x0005;
const unsigned RAND48_ADD = 0x000b;

void dorand48(unsigned short xseed[3])
{
     unsigned accu;
     unsigned short temp[2];

     accu = RAND48_MULT_0 * xseed[0] + RAND48_ADD;
     temp[0] = (unsigned short) accu;
     accu >>= 16;
     accu += RAND48_MULT_0 * xseed[1] + RAND48_MULT_1 * xseed[0];
     temp[1] = (unsigned short) accu;
     accu >>= 16;
     accu += RAND48_MULT_0 * xseed[2] + RAND48_MULT_1 * xseed[1] + RAND48_MULT_2 * xseed[0];
     xseed[0] = temp[0];
     xseed[1] = temp[1];
     xseed[2] = (unsigned short) accu;
}

float erand24(unsigned short xseed[3])
{
     dorand48(xseed);
     return ldexp((float) xseed[1], -32) + ldexp((float) xseed[2], -16);
}

const short DIFF = 0;
const short SPEC = 1;
const short REFR = 2;

float4 computeRadiance(
	Ray ray,
	__global const float *radius,
    __global const float4 *position,
    __global const float4 *emission,
    __global const float4 *color,
    __global short *reflType,
    int numSpheres,
	unsigned short Xi[3])
{
	float4 resE = (float4)(0);
	float4 resF = (float4)(1);

	for (int depth = 2; depth; --depth)
	{
		float t;
		int id;

		if (!intersectSpheres(&ray, position, radius, numSpheres, &t, &id))
		{
			resF = (float4)(0);
		}

		ray.orig += ray.dir * t;
        float4 n = normalize(ray.orig - position[id]);
        float4 nl = dot(n, ray.dir) < 0 ? n : -n;
        float4 f = color[id];

		if (reflType[id] == DIFF)
		{
            float r1 = 2 * 3.1415 * erand24(Xi);
			float r2 = erand24(Xi);
			float r2s = sqrt(r2);
            float4 w = nl;
			float4 u = normalize(cross(fabs(w.x) > 0.1 ? (float4)(0, 1, 0, 0) : (float4)(1, 0, 0, 0), w));
			float4 v = cross(w, u);

			ray.dir = normalize(u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2));

            resE += emission[id] * resF;
            resF *= f;
		}
		else if (reflType[id] == SPEC)
		{
			float4 d = ray.dir - n * (-2 * dot(n, ray.dir));
			ray.dir = d;

            resE += emission[id] * resF;
            resF *= f;
		}
		else { // REFR
			resF = (float4)(0);
		}

		ray.orig += ray.dir * 0.4;
	}

	return resE;
}

__kernel void radiance(
    __global const float *radius,
    __global const float4 *position,
    __global const float4 *emission,
    __global const float4 *color,
    __global short *reflType,
    int numSpheres,
    __global float4 *out,
    int width, int height)
{
    // get index into global data array
    int index = get_global_id(0);

    // bound check (equivalent to the limit on a 'for' loop for standard/serial C code
    if (index >= width * height)
    {   
        return; 
    }
    
	int x = index % width;
	int y = (height - index / width - 1);
	int samples = 5;
	unsigned short Xi[3] = { x * y * y * x, x * y, x * y * y };
	
	float4 outColor = (float4)(0, 0, 0, 0);
	
	float4 camOrig = (float4)(50, 52, 295.6, 0);
	float4 camDir = normalize((float4)(0, -0.042612, -1.0, 0));
	
    float4 cx = (float4)(width * .5135 / height, 0, 0, 0);
	float4 cy = normalize(cross(cx, camDir)) * .5135;

	for (int sy = 0; sy < 2; sy++) // 2x2 subpixel rows
	{
		for (int sx = 0; sx < 2; sx++) // 2x2 subpixel cols
		{
			for (int s = 0; s < samples; s++)
			{
				float r1 = 2 * erand24(Xi);
				float dx = r1 < 1 ? sqrt(r1) - 1 : (1 - sqrt(2 - r1));
                float r2 = 2 * erand24(Xi);
				float dy = r2 < 1 ? sqrt(r2) - 1 : (1 - sqrt(2 - r2));

                float4 dir = cx * (((sx + 0.5 + dx) * 0.5 + x) / width - 0.5) + cy * (((sy + 0.5 + dy) * 0.5 + y) / height - 0.5) + camDir;
             
				Ray r;
				r.orig = camOrig + dir * 140;
				r.dir = normalize(dir);
				
				outColor += computeRadiance(r, radius, position, emission, color, reflType, numSpheres, Xi) * (1.0 / samples);
			}
		}
	}
		
    out[index] = outColor * 0.25;
}
