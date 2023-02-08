import numpy as np

class Convolutionlayer:
    def __init__(self, filter_size, num_channels, num_filters, padding=0, stride=1, learning_rate=0.01):
        self.filter_size = filter_size
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.stride = stride
        self.padding = padding
        self.stride = stride
        self.learning_rate = learning_rate

        self.filters = np.random.randn(num_filters, num_channels, filter_size, filter_size)*np.sqrt(2/(num_filters+num_channels*filter_size*filter_size))
        self.bias = np.zeros(num_filters)

    def forward(self, input):
        self.input = input
        self.num_images, self.input_channels, self.input_height, self.input_width = input.shape

        if self.padding != 0:
            self.input = np.pad(self.input, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)

        self.output_height = int((self.input_height - self.filter_size + 2 * self.padding) / self.stride + 1)
        self.output_width = int((self.input_width - self.filter_size + 2 * self.padding) / self.stride + 1)

        self.input_regions = np.lib.stride_tricks.as_strided(self.input, \
            shape=(self.num_images, self.input_channels, self.output_height, self.output_width, self.filter_size, self.filter_size), \
                strides=(self.input.strides[0], self.input.strides[1], self.input.strides[2] * self.stride, self.input.strides[3] * self.stride, self.input.strides[2], self.input.strides[3]))

        self.output = np.einsum('bcijkl, nckl->bnij', self.input_regions, self.filters) + self.bias[None,:,None,None]

        return self.output

    
    def backward(self, dL_dout, outfile):
        dL_dout_changed = dL_dout

        padding = self.filter_size - 1 if self.padding == 0 else self.padding
        dilate = self.stride - 1

        if dilate != 0:
            dL_dout_changed = np.insert(dL_dout_changed, range(1,dL_dout.shape[2]), 0, axis=2)
            dL_dout_changed = np.insert(dL_dout_changed, range(1,dL_dout.shape[3]), 0, axis=3)

        if padding != 0:
            dL_dout_changed = np.pad(dL_dout_changed, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant', constant_values=0)

        dL_dout_regions = np.lib.stride_tricks.as_strided(dL_dout_changed, shape=(self.num_images, self.num_filters, self.input_height, self.input_width, self.filter_size, self.filter_size), \
            strides=(dL_dout_changed.strides[0], dL_dout_changed.strides[1], dL_dout_changed.strides[2], dL_dout_changed.strides[3], dL_dout_changed.strides[2], dL_dout_changed.strides[3]))
        rotated_filters = np.rot90(self.filters, 2, (2,3))

        dL_db = np.sum(dL_dout, axis=(0,2,3))
        dL_dfilters = np.einsum('bcijkl, bnij->nckl', self.input_regions, dL_dout)
        dL_dinput = np.einsum('bnijkl, nckl->bcij', dL_dout_regions, rotated_filters)

        outfile.write('Convolution backward\n')
        # outfile.write('self.input: ' + str(self.input) + '\n')
        outfile.write('self.filters: ' + str(self.filters) + '\n')
        # outfile.write('self.bias: ' + str(self.bias) + '\n')
        # outfile.write('self.output: ' + str(self.output) + '\n')
        # # outfile.write('dL_dout: ' + str(dL_dout.shape) + '\n')
        # # outfile.write('dL_db: ' + str(dL_db.shape) + '\n')
        # # outfile.write('dL_dfilters: ' + str(dL_dfilters.shape) + '\n')
        # # outfile.write('dL_dinput: ' + str(dL_dinput.shape) + '\n')

        self.filters -= self.learning_rate * dL_dfilters
        self.bias -= self.learning_rate * dL_db

        outfile.write('self.filters: ' + str(self.filters) + '\n')
        # outfile.write('self.bias: ' + str(self.bias) + '\n')

        return dL_dinput
        

        