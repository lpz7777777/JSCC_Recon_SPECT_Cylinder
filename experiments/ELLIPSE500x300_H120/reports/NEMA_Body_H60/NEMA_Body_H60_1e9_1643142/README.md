# NEMA H60: verified six-channel 1e9 result

See analysis.json and iteration_metrics.csv for reproducible methods and 200-frame metrics. All figures retain the ellipse FOV, use gray_r (white low, black high), sigma=0 and no edge crop. Color limits are fixed at 0..10. Background normalization is fixed per channel across iterations; values above 10 saturate visually but remain unchanged in metrics. Composite truth includes gamma yields, and is not parent 225Ac activity.

![Six-channel iteration gallery](iterations_z20.png)

![Final multiplanar truth comparison](final_multiplanar.png)

![Central two-energy detail](central_detail.png)

![CRC CNR CV curves](crc_cnr_cv_iterations.png)
