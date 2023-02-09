import numpy as np

class Pooling:
    def __init__(self, filter_size, stride=1, pool='max'):
        self.filter_size = filter_size
        self.stride = stride
        self.pool = pool

    def forward(self, input):
        self.input = input
        self.num_images, self.num_channels, self.input_height, self.input_width = input.shape

        self.output_height = int((self.input_height - self.filter_size) / self.stride + 1)
        self.output_width = int((self.input_width - self.filter_size) / self.stride + 1)
        
        self.input_regions = np.lib.stride_tricks.as_strided(input,\
            shape=(self.num_images, self.num_channels, self.output_height, self.output_width, self.filter_size, self.filter_size), \
            strides=(input.strides[0], input.strides[1], input.strides[2]*self.stride, input.strides[3]*self.stride, input.strides[2], input.strides[3]))

        if self.pool == 'max':
            self.output = np.max(self.input_regions, axis=(4,5))
        elif self.pool == 'avg':
            self.output = np.mean(self.input_regions, axis=(4,5))

        return self.output


    def backward(self, dL_dout):
        self.dL_dinput = np.zeros(self.input.shape)

        for i in range(self.output_height):
            for j in range(self.output_width):
                if self.pool == 'max':
                    data_slice = self.input[:, :, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size]
                    max_val = np.max(data_slice, axis=(2,3), keepdims=True)
                    mask = (data_slice == max_val).astype(int)
                    self.dL_dinput[:, :, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size] += mask * dL_dout[:, :, i:i+1, j:j+1]
                elif self.pool == 'avg':
                    mask = np.ones((self.num_images, self.num_channels, self.filter_size, self.filter_size))
                    self.dL_dinput[:, :, i*self.stride:i*self.stride+self.filter_size, j*self.stride:j*self.stride+self.filter_size] += mask*dL_dout[:, :, i:i+1, j:j+1] / (self.filter_size * self.filter_size)

        return self.dL_dinput