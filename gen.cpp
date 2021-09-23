#include <cmath>

#include <Halide.h>

using namespace Halide;

struct Filters2D : Generator<Filters2D> {
    // Params.

    GeneratorInput<Buffer<float>> arr{"arr", 2};
    GeneratorInput<float> scale{"scale"};
    GeneratorOutput<Buffer<float>> out{"out", 3};

    // Gaussian kernels.

    Func g0_raw{"g0_raw"};
    Func g0_sum{"g0_sum"};
    Func g0{"g0"};

    Func g1_raw{"g1_raw"};
    Func g1_sum{"g1_sum"};
    Func g1{"g1"};

    Func g2_raw{"g2_raw"};
    Func g2_sum{"g2_sum"};
    Func g2_dc{"g2_dc"};
    Func g2_centered{"g2_centered"};
    Func g2{"g2"};

    // Functions.

    Func in{"in"};

    Func gx{"gx"};
    Func gy{"gy"};

    Func dx{"dx"};
    Func dy{"dy"};

    Func dxx{"dxx"};
    Func dyy{"dyy"};
    Func dxy{"dxy"};

    Func gaussian{"gaussian"};
    Func gradmag{"gradmag"};
    Func laplacian{"laplacian"};

    // Main vars.

    Var c{"c"};
    Var x{"x"};
    Var y{"y"};
    Var z{"z"};

    // Aux vars.

    Var xi{"xi"};
    Var yi{"yi"};
    Var zi{"zi"};

    Var xo{"xo"};
    Var yo{"yo"};
    Var zo{"zo"};

    Func fx{"fx"};
    Func fy{"fy"};

    void generate() {
        Expr radius0 = cast<int>(ceil(3.0f * scale));
        Expr radius1 = cast<int>(ceil(3.5f * scale));
        Expr radius2 = cast<int>(ceil(4.0f * scale));

        RDom r0{-radius0, 2 * radius0 + 1, "r0"};
        RDom r1{-radius1, 2 * radius1 + 1, "r1"};
        RDom r2{-radius2, 2 * radius2 + 1, "r2"};

        // Gaussian kernel, and it's first and second derivatives.

        float invsqrt2pi = 1 / std::sqrt(2 * std::acos(-1));
        Expr scale2 = scale * scale;
        Expr ga0 = invsqrt2pi / scale;
        Expr ga1 = -ga0 / scale2;
        Expr ga2 = -ga1 / scale2;
        Expr gb = -0.5f / scale2;

        g0_raw(x) = ga0 * exp(gb * x * x);
        g0_sum() = sum(g0_raw(r0));
        g0(x) = g0_raw(x) / g0_sum();

        g1_raw(x) = ga1 * x * exp(gb * x * x);
        g1_sum() = abs(sum(g1_raw(r1) * r1));
        g1(x) = g1_raw(x) / g1_sum();

        g2_raw(x) = (ga1 + ga2 * x * x) * exp(gb * x * x);
        g2_dc() = sum(g2_raw(r2)) / r2.x.extent();
        g2_centered(x) = g2_raw(x) - g2_dc();
        g2_sum() = sum(g2_centered(r2) * r2 * r2) / 2;
        g2(x) = g2_centered(x) / g2_sum();

        // Functions.

        in = BoundaryConditions::mirror_interior(arr);
        // in.rename(_0, x);
        // in.rename(_1, y);

        fx(x, y) = {
            sum(g0(r0) * in(x + r0, y)),
            sum(g1(r1) * in(x + r1, y)),
            sum(g2(r2) * in(x + r2, y)),
        };

        fy(x, y) = {
            sum(g0(r0) * fx(x, y + r0)[0]),
            hypot(sum(g0(r0) * fx(x, y + r0)[1]), sum(g1(r1) * fx(x, y + r1)[0])),
            sum(g0(r0) * fx(x, y + r0)[2]) + sum(g2(r2) * fx(x, y + r2)[0]),
        };

        out(c, x, y) = mux(c, fy(x, y));
    }

    void schedule() {
        const int vec = 64;
        const int n = 3;

        out.split(x, xo, xi, vec).reorder(c, xi, y, xo).bound(c, 0, n).unroll(c).vectorize(xi);
        fx.compute_at(out, xo).vectorize(x, vec);
        in.store_root().compute_at(fx, x);

        kernels_compute_root(true);

        out.print_loop_nest();
    }

    void kernels_compute_root(bool intermediate) {
        g0.compute_root();
        g1.compute_root();
        g2.compute_root();

        if (intermediate) {
            g0_raw.compute_root();
            g0_sum.compute_root();

            g1_raw.compute_root();
            g1_sum.compute_root();

            g2_raw.compute_root();
            g2_dc.compute_root();
            g2_centered.compute_root();
            g2_sum.compute_root();
        }
    }
};

HALIDE_REGISTER_GENERATOR(Filters2D, filters2d)
