An outlier detection and recovery method for manifold-based function approximation data

Run each script from its own directory in MATLAB R2024a.
Run scripts with their own function folders. The Zhao race-car scripts share
4.3_Rotating_race_car/function. Avoid adding all folders to the MATLAB path.

4.1 Helix
  4.1.1_Comparison/our: LXX100.m and LXX100_noise.m
  4.1.1_Comparison/zhao: LXX100.m
  4.1.1_Comparison/zheng: testforluoxuanxian_100.m
  4.1.2_Parameter_sensitivity/our: LXX100_sensitivity.m

4.2 Klein bottle
  4.2.1_Comparison/our: KLB100.m and KB100_noise.m
  4.2.1_Comparison/zhao: KLB100.m
  4.2.1_Comparison/zheng: KleinBottle_100.m
  4.2.2_Function_approximation/our: KB_function_approximation.m

4.3 Rotating race car
  4.3.1_Comparison/our: Run car100.m before carEpstu.m.
  4.3.1_Comparison/zhao: car100.m
  4.3.2_Computational_cost/our and zhao: car100neicun_and_time.m
  Images are bundled in data/car. Working-set measurements use Windows.

For each experimental setting, we initialized the random number generator with a fixed seed. Within each trial, all methods under comparison received the same outlier realization, with identical outlier locations, magnitudes, and signs.

Our method builds the approximation matrix with buildLocalPcaMLS(X, d, m),
where X has one sample per row. The fifth output Btest maps training
function values to independent test points in the approximation experiment.
Use MMLSforoutlier(A, f, U, UN, tau, eta) to estimate the outlier deviations.
The default OMP ratio threshold is 0.5 if tau is omitted or empty.
Before adding multiplicative noise, the robustness scripts inject outliers.
