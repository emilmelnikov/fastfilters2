#include <cmath>

#include <Halide.h>

using namespace Halide;

struct Filters2D : Generator<Filters2D> {
    GeneratorInput<Buffer<float>> arr{"arr", 2};
    GeneratorInput<Buffer<float>> k0{"k0", 1};
    GeneratorInput<Buffer<float>> k1{"k1", 1};
    GeneratorInput<Buffer<float>> k2{"k2", 1};
    GeneratorOutput<Buffer<float>> out{"out", 3};

    Func in{"in"};
    Func fx{"fx"};
    Func fy{"fy"};

    Func gaussian{"gaussian"};
    Func gradmag{"gradmag"};
    Func laplacian{"laplacian"};

    Var c{"c"};
    Var x{"x"};
    Var y{"y"};

    Var xi{"xi"};
    Var yi{"yi"};

    Var xo{"xo"};
    Var yo{"yo"};

    void generate() {
        RDom r0{k0};
        RDom r1{k1};
        RDom r2{k2};

        in = BoundaryConditions::mirror_interior(arr);
        // in.rename(_0, x);
        // in.rename(_1, y);

        fx(x, y) = {
            sum(k0(r0) * in(x + r0, y)),
            sum(k1(r1) * in(x + r1, y)),
            sum(k2(r2) * in(x + r2, y)),
        };

        fy(x, y) = {
            sum(k0(r0) * fx(x, y + r0)[0]),
            hypot(sum(k0(r0) * fx(x, y + r0)[1]), sum(k1(r1) * fx(x, y + r1)[0])),
            sum(k0(r0) * fx(x, y + r0)[2]) + sum(k2(r2) * fx(x, y + r2)[0]),
        };

        out(c, x, y) = mux(c, fy(x, y));
    }

    void schedule() {
        const int vec = 64;
        const int n = 3;

        out.split(x, xo, xi, vec).reorder(c, xi, y, xo).bound(c, 0, n).unroll(c).vectorize(xi);
        fx.compute_at(out, xo).vectorize(x, vec);
        in.store_root().compute_at(fx, x);

        out.print_loop_nest();
    }
};

HALIDE_REGISTER_GENERATOR(Filters2D, filters2d)
