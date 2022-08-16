

```cpp
#ifndef vtkImageResize_h
#define vtkImageResize_h

#include "vtkImagingCoreModule.h" // For export macro
#include "vtkThreadedImageAlgorithm.h"

class vtkAbstractImageInterpolator;

class VTKIMAGINGCORE_EXPORT vtkImageResize : public vtkThreadedImageAlgorithm
{
public:
  static vtkImageResize *New();
  vtkTypeMacro(vtkImageResize, vtkThreadedImageAlgorithm);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  enum
  {
    OUTPUT_DIMENSIONS,
    OUTPUT_SPACING,
    MAGNIFICATION_FACTORS
  };
```



















