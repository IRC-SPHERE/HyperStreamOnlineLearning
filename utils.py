import numpy as np

def generate_hidden_images(generator, digit_size=8, n=15, minmax=15):
    image = np.zeros((digit_size * n, digit_size * n))
    # we will sample n points within [-15, 15] standard deviations
    grid_x = np.linspace(-minmax, minmax, n)
    grid_y = np.linspace(-minmax, minmax, n)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.inverse_transform(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            image[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit
    return image
