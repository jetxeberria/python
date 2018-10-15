import pycuda.driver as cuda
import pycuda.autoinit		# Autoruns. Runs initialization, context
				# creation and cleanup
from pycuda.compiler import SourceModule

# Import basic arrays library
import numpy

a = numpy.random.randn(4,4)

# Cast values to single-precision numbers.
a = a.astype(numpy.float32)

# Allocate memory on the device
a_gpu = cuda.mem_alloc(a.nbytes)

# Transfer data to GPU
cuda.memcpy_htod(a_gpu, a)

# Creating a kernel. Write the corresponding CUDA C code, and feed it into the
# constructor of a pycuda.compiler.SourceModule
mod = SourceModule("""
	__global__ void doublify(float *a)
	{
		int idx = threadIdx.x + threadIdx.y*4;
		a[idx] *= 2;

		if (idx == 0)
		{
			printf("One wrap along GPU.");
			//printf("threadIdx.x: %d threadIdx.y: %d", threadIdx.x, threadIdx.y);
		}
	}
	""")

func = mod.get_function("doublify")

# Call to that reference with proper argument
func(a_gpu, block=(4,4,1))

# Fetch the data back from the GPU and display it 
# Allocate memory space for CPU variable 'a_doubled'
a_doubled = numpy.empty_like(a)

# Copy memory from GPU to CPU
cuda.memcpy_dtoh(a_doubled, a_gpu)

# Set properly print settings
numpy.set_printoptions(precision=3, suppress=True)

# Print results
print '\na_doubled: {}'.format(a_doubled)
print 'a: {}'.format(a)
print 'a_gpu: {}'.format(a_gpu)
