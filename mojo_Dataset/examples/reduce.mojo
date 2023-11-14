# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# This sample implements a simple reduction operation on a
# large array of values to produce a single result.
# Reductions and scans are common algorithm patterns in parallel computing.

import benchmark
from time import now
from algorithm import sum
from random import rand
from memory.buffer import Buffer
from python import Python

# Change these numbers to reduce on different sizes
alias size_small: Int = 1 << 21
alias size_large: Int = 1 << 27

# Datatype for Tensor/Array
alias type = DType.float32


# Use the https://en.wikipedia.org/wiki/Kahan_summation_algorithm
# Simple summation of the array elements
fn naive_reduce_sum[size: Int](array: Tensor[type]) -> Float32:
    let A = array
    var my_sum = array[0]
    var c: Float32 = 0.0
    for i in range(array.dim(0)):
        let y = array[i] - c
        let t = my_sum + y
        c = (t - my_sum) - y
        my_sum = t
    return my_sum


fn stdlib_reduce_sum[size: Int](array: Tensor[type]) -> Float32:
    let my_sum = sum(array._to_buffer())
    return my_sum


fn pretty_print(name: StringLiteral, elements: Int, time: Float64) raises:
    let py = Python.import_module("builtins")
    _ = py.print(
        py.str("{:<16} {:>11,} {:>8.2f}ms").format(
            String(name) + " elements:", elements, time
        )
    )


fn bench[
    func: fn[size: Int] (array: Tensor[type]) -> Float32,
    size: Int,
    name: StringLiteral,
](array: Tensor[type]) raises:
    @parameter
    fn runner():
        let result = func[size](array)
        benchmark.keep(result)

    let ms = benchmark.run[runner]().mean[benchmark.Unit.ms]()
    pretty_print(name, size, ms)


fn main() raises:
    print(
        "Sum all values in a small array and large array\n"
        "Shows algorithm.sum from stdlib with much better scaling\n"
    )
    # Create two 1-dimensional tensors i.e. arrays
    let small_array = rand[type](size_small)
    let large_array = rand[type](size_large)

    bench[naive_reduce_sum, size_small, "naive"](small_array)
    bench[naive_reduce_sum, size_large, "naive"](large_array)

    bench[stdlib_reduce_sum, size_small, "stdlib"](small_array)
    bench[stdlib_reduce_sum, size_large, "stdlib"](large_array)
