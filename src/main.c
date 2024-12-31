#include "gaussian_blur.c"
#include <assert.h>

int main(int argc, char* argv[])
{
    Parameters params = parse_args(argc, argv);
    assert(validate_parameters(&params));
    Image image = read_image(params.image_filepath);
    Image diffused_image = apply_gaussian_blur(&image, &params);
    write_image(diffused_image, params.output_filepath);
    return 0;
}
