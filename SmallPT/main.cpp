#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008 
#include <stdlib.h> // Make : g++ -O3 -fopenmp smallpt.cpp -o smallpt 
#include <stdio.h>  //        Remove "-fopenmp" for g++ version < 4.2 

unsigned RAND48_SEED_0  (0x330e);
unsigned RAND48_SEED_1  (0xabcd);
unsigned RAND48_SEED_2  (0x1234);
unsigned RAND48_MULT_0  (0xe66d);
unsigned RAND48_MULT_1  (0xdeec);
unsigned RAND48_MULT_2  (0x0005);
unsigned RAND48_ADD     (0x000b);

unsigned short _rand48_seed[3] = {
     RAND48_SEED_0,
     RAND48_SEED_1,
     RAND48_SEED_2
};
unsigned short _rand48_mult[3] = {
     RAND48_MULT_0,
     RAND48_MULT_1,
     RAND48_MULT_2
};

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

double erand48(unsigned short xseed[3])
{
     dorand48(xseed);
     return ldexp((double) xseed[0], -48) +
            ldexp((double) xseed[1], -32) +
            ldexp((double) xseed[2], -16);
}

float erand24(unsigned short xseed[3])
{
     dorand48(xseed);
     return ldexp((float) xseed[1], -32) + ldexp((float) xseed[2], -16);
}

struct Vec
{
     float x, y, z;
     Vec(float x_ = 0, float y_ = 0, float z_ = 0)
     {
          x = x_;
          y = y_;
          z = z_;
     }
     Vec operator+(const Vec &b) const
     {
          return Vec(x + b.x, y + b.y, z + b.z);
     }
     Vec operator-(const Vec &b) const
     {
          return Vec(x - b.x, y - b.y, z - b.z);
     }

     Vec operator-() const
     {
         return Vec(-x, -y, -z);
     }

     Vec operator*(float b) const {
          return Vec(x * b, y * b, z * b);
     }
     Vec mult(const Vec &b) const {
          return Vec(x * b.x, y * b.y, z * b.z);
     }
     Vec& norm() {
          return *this = *this * (1 / sqrt(x * x + y * y + z * z));
     }
     float dot(const Vec &b) const {
          return x * b.x + y * b.y + z * b.z;     // cross:
     }
     Vec operator%(Vec&b) {
          return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x);
     }
};

struct Ray
{
     Vec o, d;
     Ray(Vec o_, Vec d_) : o(o_), d(d_) {}
};

enum Refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()
struct Sphere
{
    float rad;       // radius
    Vec p, e, c;      // position, emission, color
    Refl_t refl;      // reflection type (DIFFuse, SPECular, REFRactive)

    Sphere(float rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) : rad(rad_), p(p_), e(e_), c(c_), refl(refl_) { }

    float intersect(const Ray &r) const // returns distance, 0 if nohit
    {  
        Vec op = p - r.o; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
        double t, eps = 1e-4, b = op.dot(r.d), det = b * b - op.dot(op) + rad * rad;
        if (det < 0) return 0;
        else det = sqrt(det);
        return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
    }
};

Sphere spheres[] = {//Scene: radius, position, emission, color, material
     Sphere(1e5, Vec( 1e5 + 1, 40.8, 81.6), Vec(), Vec(.75, .25, .25), DIFF), //Left
     Sphere(1e5, Vec(-1e5 + 99, 40.8, 81.6), Vec(), Vec(.25, .25, .75), DIFF), //Rght
     Sphere(1e5, Vec(50, 40.8, 1e5),     Vec(), Vec(.75, .75, .75), DIFF), //Back
     Sphere(1e5, Vec(50, 40.8, -1e5 + 170), Vec(), Vec(),           DIFF), //Frnt
     Sphere(1e5, Vec(50, 1e5, 81.6),    Vec(), Vec(.75, .75, .75), DIFF), //Botm
     Sphere(1e5, Vec(50, -1e5 + 81.6, 81.6), Vec(), Vec(.75, .75, .75), DIFF), //Top
     Sphere(16.5, Vec(27, 16.5, 47),       Vec(), Vec(1, 1, 1)*.999, SPEC), //Mirr
     Sphere(16.5, Vec(73, 16.5, 78),       Vec(), Vec(1, 1, 1)*.999, REFR), //Glas
     Sphere(600, Vec(50, 681.6 - .27, 81.6), Vec(12, 12, 12),  Vec(), DIFF) //Lite
};

inline float clamp(float x)
{
     return x < 0 ? 0 : x > 1 ? 1 : x;
}
inline int toInt(float x)
{
     return int(pow(clamp(x), 1 / 2.2f) * 255 + .5f);
}

bool intersect(const Ray& r, double& t, int& id)
{
    double inf = t = 1e20;
    for(int i = sizeof(spheres) / sizeof(Sphere); i--;)
    {
        double d = spheres[i].intersect(r);
        if (d && d < t) {
            t = d;
            id = i;
        }
    }
    return t < inf;
}

struct RadState
{
    Vec e;
    Vec f;
};


Vec radiance(Ray r, int depth, unsigned short *Xi)
{
    Vec resE(0, 0, 0);
    Vec resF(1, 1, 1);

    for (int depth = 0; depth != 10; ++depth)
    {
        double t;                               // distance to intersection
        int id = 0;                               // id of intersected object

        if (!intersect(r, t, id))
        {
            resF = Vec();
            continue;
        }

        const Sphere &obj = spheres[id];        // the hit object

        Vec x = r.o + r.d * t;
        Vec n = (x - obj.p).norm();
        Vec nl = n.dot(r.d) < 0 ? n : -n;
        Vec f = obj.c;

        double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl

        if (obj.refl == DIFF)
        {                 // Ideal DIFFUSE reflection
            double r1 = 2 * M_PI * erand48(Xi), r2 = erand48(Xi), r2s = sqrt(r2);
            Vec w = nl, u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(), v = w % u;
            Vec d = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();

            r = Ray(x + d * 0.1, d);
            resE = resE + obj.e.mult(resF);
            resF = resF.mult(f);
            continue;
        }
        else if (obj.refl == SPEC)
        {// Ideal SPECULAR reflection
            r = Ray(x, r.d - n * 2 * n.dot(r.d));
            resE = resE + obj.e.mult(resF);
            resF = resF.mult(f);
            continue;
        }
        else { 
            resF = Vec();
            continue;
        }

        Ray reflRay(x, r.d - n * 2 * n.dot(r.d)); // Ideal dielectric REFRACTION
        bool into = n.dot(nl) > 0;              // Ray from outside going in?
        double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = r.d.dot(nl), cos2t;
        if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0 && depth < 5) // Total internal reflection
        {
            r = reflRay;
            resE = resE + obj.e.mult(resF);
            resF = resF.mult(f);
            continue;
        }

        Vec tdir = (r.d * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
        double a = nt - nc;
        double b = nt + nc;
        double R0 = a * a / (b * b);
        double c = 1 - (into ? -ddn : tdir.dot(n));
        double Re = R0 + (1 - R0) * c * c * c * c * c;
        double Tr = 1 - Re;
        double P = .25 + .5 * Re;
        double RP = Re / P;
        double TP = Tr / (1 - P);
    
        if (erand48(Xi) < P)
        {
            r = reflRay;
            resE = resE + obj.e.mult(resF) * RP;
            resF = resF.mult(f) * RP;
        }
        else
        {
            r = Ray(x, tdir);
            resE = resE + obj.e.mult(resF) * TP;
            resF = resF.mult(f) * TP;
        }
    }

    return resE;
}
/*
 Vec radiance(const Ray &r, int depth, unsigned short *Xi){ 
   double t;                               // distance to intersection 
   int id=0;                               // id of intersected object 
   if (!intersect(r, t, id)) return Vec(); // if miss, return black 
   const Sphere &obj = spheres[id];        // the hit object 
   Vec x=r.o+r.d*t, n=(x-obj.p).norm(), nl=n.dot(r.d)<0?n:n*-1, f=obj.c; 
   double p = f.x>f.y && f.x>f.z ? f.x : f.y>f.z ? f.y : f.z; // max refl 
   if (++depth>5) if (erand48(Xi)<p) f=f*(1/p); else return obj.e; //R.R. 
   if (obj.refl == DIFF){                  // Ideal DIFFUSE reflection 
     double r1=2*M_PI*erand48(Xi), r2=erand48(Xi), r2s=sqrt(r2); 
     Vec w=nl, u=((fabs(w.x)>.1?Vec(0,1):Vec(1))%w).norm(), v=w%u; 
     Vec d = (u*cos(r1)*r2s + v*sin(r1)*r2s + w*sqrt(1-r2)).norm(); 
     return obj.e + f.mult(radiance(Ray(x,d),depth,Xi)); 
   } else if (obj.refl == SPEC)            // Ideal SPECULAR reflection 
     return obj.e + f.mult(radiance(Ray(x,r.d-n*2*n.dot(r.d)),depth,Xi)); 
   Ray reflRay(x, r.d-n*2*n.dot(r.d));     // Ideal dielectric REFRACTION 
   bool into = n.dot(nl)>0;                // Ray from outside going in? 
   double nc=1, nt=1.5, nnt=into?nc/nt:nt/nc, ddn=r.d.dot(nl), cos2t; 
   if ((cos2t=1-nnt*nnt*(1-ddn*ddn))<0)    // Total internal reflection 
     return obj.e + f.mult(radiance(reflRay,depth,Xi)); 
   Vec tdir = (r.d*nnt - n*((into?1:-1)*(ddn*nnt+sqrt(cos2t)))).norm(); 
   double a=nt-nc, b=nt+nc, R0=a*a/(b*b), c = 1-(into?-ddn:tdir.dot(n)); 
   double Re=R0+(1-R0)*c*c*c*c*c,Tr=1-Re,P=.25+.5*Re,RP=Re/P,TP=Tr/(1-P); 
   return
     obj.e + f.mult(
     (erand48(Xi)<P ?   // Russian roulette 
        radiance(reflRay,depth,Xi)*RP:
            radiance(Ray(x,tdir),depth,Xi)*TP)); 
 }
 */
int main(int argc, char *argv[])
{
     int w = 320, h = 240, samps = 20; // # samples
     Ray cam(Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm()); // cam pos, dir
     Vec cx = Vec(w * .5135 / h), cy = (cx % cam.d).norm() * .5135, r, *c = new Vec[w*h];
#pragma omp parallel for schedule(dynamic, 1) private(r)       // OpenMP 
     for (int y = 0; y < h; y++) {                  // Loop over image row s
          fprintf(stderr, "\rRendering (%d spp) %5.2f%%", samps * 4, 100.*y / (h - 1));
          unsigned short Xi[3] = {0, 0, y*y*y};
          for (unsigned short x = 0; x < w; x++) // Loop cols
               for (int sy = 0, i = (h - y - 1) * w + x; sy < 2; sy++) // 2x2 subpixel rows
                    for (int sx = 0; sx < 2; sx++, r = Vec()) { // 2x2 subpixel cols
                         for (int s = 0; s < samps; s++) {
                              double r1 = 2 * erand48(Xi), dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
                              double r2 = 2 * erand48(Xi), dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
                              Vec d = cx * ( ( (sx + .5 + dx) / 2 + x) / w - .5) +
                                      cy * ( ( (sy + .5 + dy) / 2 + y) / h - .5) + cam.d;
                              r = r + radiance(Ray(cam.o + d * 140, d.norm()), 0, Xi) * (1. / samps);
                         } // Camera rays are pushed ^^^^^ forward to start in interior
                         c[i] = c[i] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
                    }
     }
     FILE *f = fopen("image.ppm", "w");         // Write image to PPM file.
     fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
     for (int i = 0; i < w * h; i++)
          fprintf(f, "%d %d %d ", toInt(c[i].x), toInt(c[i].y), toInt(c[i].z));
}