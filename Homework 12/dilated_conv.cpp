#include "Halide.h"
#include "common.h"

#include <stdio.h>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    const int N = 5, CI = 128, CO = 128, W = 100, H = 80, KW = 3, KH = 3;
    const int dilation = 15;

    ImageParam input(type_of<float>(), 4);
    ImageParam filter(type_of<float>(), 4);

    // Define variables and reduction domain
    Var x("x"), y("y"), c("c"), n("n");
    Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
    Var co("co"), ci("ci");

    Func dilated_conv("dilated_conv");
    RDom r(0, CI, 0, KW, 0, KH);

    // Algorithm definition
    dilated_conv(c, x, y, n) = 0.0f;
    dilated_conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * 
        input(r.x, x + r.y * (dilation + 1), y + r.z * (dilation + 1), n);

    // **Scheduling**
    // 1. Split the x and y dimensions for tiling
    dilated_conv.compute_root()
        .tile(x, y, xo, yo, xi, yi, 8, 8)  // 8x8 tile size (tunable)
        .fuse(xo, yo, co)
        .parallel(co)                     // Parallelize outer loop over tiles
        .vectorize(xi, 8);                // Vectorize inner x dimension

    // 2. Optimize the reduction
    dilated_conv.update()
        .reorder(r.x, r.y, r.z, c, xi, yi, n)
        .unroll(r.z)                      // Unroll kernel height loop
        .unroll(r.y)                      // Unroll kernel width loop
        .vectorize(xi, 8)                 // Vectorize inner computation
        .parallel(n);                     // Parallelize across batch dimension

    // Buffer initialization
    Buffer<float, 4> in(CI, W + (KW - 1) * (dilation + 1), H + (KH - 1) * (dilation + 1), N);
    Buffer<float, 4> fil(CO, KW, KH, CI);
    Buffer<float, 4> output_halide(CO, W, H, N);

    // Initialize input and filter with random data
    random_data<float, 4>(in);
    random_data<float, 4>(fil);
    input.set(in);
    filter.set(fil);

    // JIT compile and run
    dilated_conv.realize(output_halide);
    double t_halide = benchmark(10, 10, [&]() { dilated_conv.realize(output_halide); });

    Buffer<float, 4> output_ref(CO, W, H, N);
    double t_onednn = dnnl_dilated_conv_wrapper(in.data(), fil.data(), output_ref.data(), 
                                                {N, CI, CO, W, H, KW, KH, dilation, dilation});

    // Check correctness
    if (check_equal<float, 4>(output_ref, output_halide)) {
        printf("Halide results - OK\n");
    } else {
        printf("Halide results - FAIL\n");
        return 1;
    }

    float gflops = 2.0f * (N * CO * H * W) * (CI * KH * KW) / 1e9f;
    printf("Halide: %fms, %f GFLOP/s\n", t_halide * 1e3, (gflops / t_halide));
    printf("oneDNN: %fms, %f GFLOP/s\n\n", t_onednn * 1e3, (gflops / t_onednn));

    printf("Success!\n");
    return 0;
}
