#pragma once

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define CONST_FLOAT4(ptr) (*reinterpret_cast<const float4*>(ptr))