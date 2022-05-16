# Profiling Results

Time-based profiling after porting `CMI_VE` to the GPU shows that

- 26.2% of the time is spent on `CME_VI`, down from 96.8%. The total time in this function is down to 28.3s from 974s, which is a ~34x speedup.
- 40% of the time application is now spent importing libs (boostraping)
- The remaining ~30% is spent under the main module doing some data prep.


![image](https://user-images.githubusercontent.com/84105092/168638696-054fb148-234b-4eae-b02f-5114b60a96a7.png)
