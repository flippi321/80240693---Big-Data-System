#include "Halide.h"
#include "common.h"

#include <stdio.h>

using namespace Halide;
using namespace Halide::Tools;

int main(int argc, char **argv) {
    const int N = 5, CI = 128, CO = 128, W = 100, H = 80, KW = 3, KH = 3;
    const int dilation = 63;

    ImageParam input(type_of<float>(), 4);
    ImageParam filter(type_of<float>(), 4);

    // define Halide variables
    Var x("x"), y("y"), c("c"), n("n");
    Var c_out, c_in, x_out, x_in, tile_idx;

    // Define the computation: dilated convolution
    Func dilated_conv("dilated_conv");
    RDom r(0, CI, 0, KW, 0, KH, "r");

    dilated_conv(c, x, y, n) = 0.0f;
    dilated_conv(c, x, y, n) += filter(c, r.y, r.z, r.x) * input(r.x, x + r.y * (dilation + 1), y + r.z * (dilation + 1), n);

    // Define the output function
    Func output("output");
    output(c, x, y, n) = dilated_conv(c, x, y, n);

    // Scheduling
    output.tile(c, x, c_out, x_out, c_in, x_in, 64, 4)  // Tile across channels and spatial x
          .vectorize(c_in)                              // Vectorize within tiles
          .vectorize(x_in)                              // Vectorize within spatial tiles
          .fuse(c_out, x_out, tile_idx)                 // Fuse outer tile loops for parallelism
          .parallel(tile_idx)                           // Parallelize tile traversal
          .parallel(y)                                  // Parallelize over spatial y
          .parallel(n);                                 // Parallelize over batch dimension

    // Uncomment for debugging the loop nest
    output.print_loop_nest();

    // Create and initialize input buffers
    Buffer<float, 4> in(CI, W + (KW - 1) * (dilation + 1), H + (KH - 1) * (dilation + 1), N);
    Buffer<float, 4> fil(CO, KW, KH, CI);
    Buffer<float, 4> output_halide(CO, W, H, N);

    // init randomly
    random_data<float, 4>(in);
    random_data<float, 4>(fil);
    input.set(in);
    filter.set(fil);

    // jit compile and warm-up
    dilated_conv.realize(output_halide);
    // NOTE: uncomment next line if time is unstable
    // double t_halide = benchmark(10, 10, [&]() { dilated_conv.realize(output_halide); });
    double t_halide = benchmark(1, 1, [&]() { dilated_conv.realize(output_halide); });

    Buffer<float, 4> output_ref(CO, W, H, N);
    // create and execute a dilated conv primitive using oneDNN
    double t_onednn = dnnl_dilated_conv_wrapper(in.data(), fil.data(), output_ref.data(), {N, CI, CO, W, H, KW, KH, dilation, dilation});

    // check results
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
