#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "THC/generic/THCTensorMathCompareT.cu"
#else

void THCTensor_(ltTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorLTOp<scalar_t,
                                   bool>());
}

void THCTensor_(gtTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorGTOp<scalar_t,
                                   bool>());
}

void THCTensor_(leTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorLEOp<scalar_t,
                                   bool>());
}

void THCTensor_(geTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorGEOp<scalar_t,
                                   bool>());
}

void THCTensor_(eqTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorEQOp<scalar_t,
                                   bool>());
}

void THCTensor_(neTensor)(THCState *state, THCudaBoolTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<bool, scalar_t>(state, self_, src1, src2,
                                   TensorNEOp<scalar_t,
                                   bool>());
}

void THCTensor_(ltTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorLTOp<scalar_t,
                                scalar_t>());
}

void THCTensor_(gtTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorGTOp<scalar_t,
                                scalar_t>());
}

void THCTensor_(leTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorLEOp<scalar_t,
                                scalar_t>());
}

void THCTensor_(geTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorGEOp<scalar_t,
                                scalar_t>());
}

void THCTensor_(eqTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorEQOp<scalar_t,
                                scalar_t>());
}

void THCTensor_(neTensorT)(THCState *state, THCTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<scalar_t, scalar_t>(state, self_, src1, src2,
                                TensorNEOp<scalar_t,
                                scalar_t>());
}

void THCTensor_(ltTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorLTOp<scalar_t,
                                             unsigned char>());
}

void THCTensor_(gtTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorGTOp<scalar_t,
                                             unsigned char>());
}

void THCTensor_(leTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorLEOp<scalar_t,
                                             unsigned char>());
}

void THCTensor_(geTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorGEOp<scalar_t,
                                             unsigned char>());
}

void THCTensor_(eqTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorEQOp<scalar_t,
                                             unsigned char>());
}

void THCTensor_(neTensorByte)(THCState *state, THCudaByteTensor *self_, THCTensor *src1, THCTensor *src2)
{
  THCAssertSameGPU(THCTensor_(checkGPU)(state, 3, self_, src1, src2));
  THC_logicalTensor<unsigned char, scalar_t>(state, self_, src1, src2,
                                             TensorNEOp<scalar_t,
                                             unsigned char>());
}

#endif
