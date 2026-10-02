# Population sweep at 20 dimensions

Starting populations: [128, 512, 1024, 2048, 5000]. 10 seeds per case; 100,000 evaluations allowed per run; five elites; maximum population 5,000. 1200 measured fractal runs, 0 early-stop errors, 117,087,014 actual objective evaluations.

All fractal methods use Wave. The three non-restarting variants keep the specified population. The basin controller retains its automatic population and scale schedules, so its starting population can change between rounds, up to 5,000. Initial populations are varied between independent runs, not edited during one run. Other settings remain those of the previous 20D benchmark: Gaussian/local covariance standard deviation 0.2, adaptive scale bounds [0.0001, 0.2], nonperiodic boundaries, and COCO instance 1. No optimizer changes or tuning.

CMA-ES uses its own default population and selection rule. Its unchanged 20D results are reused once as a reference, with the native-library fingerprint and evaluation budget verified. They are not relabeled as 128–5,000-walker CMA runs.

Error is best evaluated objective minus the known optimum; lower is better. Success means error ≤ 1e-6. Failed runs retain their last best objective. A dagger (†) marks groups containing errors; inspect their evaluation counts. Only complete operations are admitted, so actual counts may fall below the allowance even without an error. The known outer session guard can stop an invalid population before saved elites are restored.

At 100,000 evaluations, larger populations allow fewer movement/adaptation steps. The controller's round allowance is 50 × dimension × population and its stagnation interval is 10 × dimension × population. A controller-enabled run can therefore complete without any restart; round counts below distinguish that case from actual multi-round search.

## Median error by starting population

| Problem | Variant | 128 | 512 | 1,024 | 2,048 | 5,000 |
|---|---|---:|---:|---:|---:|---:|
| Quadratic bowl | Gaussian | 0.0159512 | 0.0294604 | 0.107305 | 2.36585 | 6.98547 |
| Quadratic bowl | Local covariance | 0.00948842 | 0.0284109 | 0.0807075 | 1.17063 | 7.18436 |
| Quadratic bowl | Bounded adaptive | 0.000126097 | 1.97139 | 6.38593 | 9.38041 | 11.0735 |
| Quadratic bowl | Bounded + basin restarts | 0.000126097 | 1.97139 | 6.38593 | 9.38041 | 11.0735 |
| Rotated ill-conditioned ellipsoid | Gaussian | 13880.3 | 27159 | 32361.2 | 86437.3 | 156474 |
| Rotated ill-conditioned ellipsoid | Local covariance | 20029.7 | 30606.4 | 40543.9 | 91365.1 | 151488 |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 8010.1 | 88487 | 129203 | 346085 | 784117 |
| Rotated ill-conditioned ellipsoid | Bounded + basin restarts | 8010.1 | 88487 | 129203 | 346085 | 784117 |
| Rastrigin | Gaussian | 79.6846 | 104.818 | 118.241 | 141.057 | 155.989 |
| Rastrigin | Local covariance | 83.1232 | 100.27 | 119.708 | 137.229 | 168.306 |
| Rastrigin | Bounded adaptive | 95.5156 | 92.9896 | 109.147 | 133.165 | 205.572 |
| Rastrigin | Bounded + basin restarts | 95.5156 | 92.9896 | 109.147 | 133.165 | 205.572 |
| Rotated Rastrigin | Gaussian | 110.802 | 130.461 | 146.05 | 203.156 | 266.773 |
| Rotated Rastrigin | Local covariance | 111.842 | 131.189 | 147.855 | 201.245 | 291.788 |
| Rotated Rastrigin | Bounded adaptive | 360.697 | 263.962 | 287.715 | 365.699 | 469.627 |
| Rotated Rastrigin | Bounded + basin restarts | 360.697 | 263.962 | 287.715 | 365.699 | 469.627 |
| Rosenbrock | Gaussian | 45.7696 | 206.371 | 1241.29 | 30854.2 | 190967 |
| Rosenbrock | Local covariance | 33.5716 | 129.956 | 669.246 | 13767.2 | 183064 |
| Rosenbrock | Bounded adaptive | 18.1796 | 11557.5 | 168479 | 392231 | 595187 |
| Rosenbrock | Bounded + basin restarts | 18.1796 | 11557.5 | 168479 | 392231 | 595187 |
| Boundary optimum (linear slope) | Gaussian | 35.9146 | 48.5692 | 68.0491 | 107.017 | 142.077 |
| Boundary optimum (linear slope) | Local covariance | 34.7661 | 52.4476 | 77.0552 | 102.64 | 137.722 |
| Boundary optimum (linear slope) | Bounded adaptive | 68.8694 | 96.8033 | 135.86 | 170.259 | 186.309 |
| Boundary optimum (linear slope) | Bounded + basin restarts | 68.8694 | 96.8033 | 135.86 | 170.259 | 186.309 |

## Unchanged CMA-ES reference

| Problem | Median error | Successes |
|---|---:|---:|
| Quadratic bowl | 1.53123e-15 | 10/10 |
| Rotated ill-conditioned ellipsoid | 7.10543e-15 | 10/10 |
| Rastrigin | 4.47732 | 0/10 |
| Rotated Rastrigin | 5.96975 | 0/10 |
| Rosenbrock | 1.879e-15 | 10/10 |
| Boundary optimum (linear slope) | 0 | 10/10 |

## Variation, accounting, and actual population

IQR uses inclusive 25th–75th percentiles. CPU time includes engine work and benchmark bookkeeping. Concurrent workers affect timings; these are not isolated latency measurements. Success counts and variability are descriptive, without significance tests.

| Problem | Variant | Initial N | Median error | IQR | Successes | Errors | Evaluations min–max | Steps median | Actual N min–max | Rounds min–max | Median CPU seconds |
|---|---|---:|---:|---|---:|---:|---|---:|---|---|---:|
| Quadratic bowl | Gaussian | 128 | 0.0159512 | 0.01549–0.01793 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 1.763 |
| Quadratic bowl | Gaussian | 512 | 0.0294604 | 0.02371–0.03148 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 1.724 |
| Quadratic bowl | Gaussian | 1024 | 0.107305 | 0.09335–0.1603 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 1.745 |
| Quadratic bowl | Gaussian | 2048 | 2.36585 | 2.221–2.606 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 1.657 |
| Quadratic bowl | Gaussian | 5000 | 6.98547 | 6.019–7.373 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 1.715 |
| Quadratic bowl | Local covariance | 128 | 0.00948842 | 0.008453–0.00989 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 19.201 |
| Quadratic bowl | Local covariance | 512 | 0.0284109 | 0.02751–0.03009 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 6.350 |
| Quadratic bowl | Local covariance | 1024 | 0.0807075 | 0.06718–0.08966 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 4.264 |
| Quadratic bowl | Local covariance | 2048 | 1.17063 | 0.8879–2.067 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 2.947 |
| Quadratic bowl | Local covariance | 5000 | 7.18436 | 6.25–7.541 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 2.405 |
| Quadratic bowl | Bounded adaptive | 128 | 0.000126097 | 8.41e-05–0.0001873 | 0/10 | 0 | 99617–99753 | 707 | 128–128 | 1–1 | 4.923 |
| Quadratic bowl | Bounded adaptive | 512 | 1.97139 | 1.465–2.42 | 0/10 | 0 | 98467–99037 | 175 | 512–512 | 1–1 | 4.967 |
| Quadratic bowl | Bounded adaptive | 1024 | 6.38593 | 4.53–7.168 | 0/10 | 0 | 97799–98050 | 86 | 1024–1024 | 1–1 | 4.998 |
| Quadratic bowl | Bounded adaptive | 2048 | 9.38041 | 8.46–10.4 | 0/10 | 0 | 94336–94551 | 41 | 2048–2048 | 1–1 | 5.101 |
| Quadratic bowl | Bounded adaptive | 5000 | 11.0735 | 10.28–12.37 | 0/10 | 0 | 87372–87691 | 15 | 5000–5000 | 1–1 | 5.435 |
| Quadratic bowl | Bounded + basin restarts | 128 | 0.000126097 | 8.41e-05–0.0001873 | 0/10 | 0 | 99617–99753 | 707 | 128–128 | 1–1 | 7.248 |
| Quadratic bowl | Bounded + basin restarts | 512 | 1.97139 | 1.465–2.42 | 0/10 | 0 | 98467–99037 | 175 | 512–512 | 1–1 | 7.759 |
| Quadratic bowl | Bounded + basin restarts | 1024 | 6.38593 | 4.53–7.168 | 0/10 | 0 | 97799–98050 | 86 | 1024–1024 | 1–1 | 8.221 |
| Quadratic bowl | Bounded + basin restarts | 2048 | 9.38041 | 8.46–10.4 | 0/10 | 0 | 94336–94551 | 41 | 2048–2048 | 1–1 | 8.346 |
| Quadratic bowl | Bounded + basin restarts | 5000 | 11.0735 | 10.28–12.37 | 0/10 | 0 | 87372–87691 | 15 | 5000–5000 | 1–1 | 8.226 |
| Rotated ill-conditioned ellipsoid | Gaussian | 128 | 13880.3 | 9788–1.499e+04 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 3.950 |
| Rotated ill-conditioned ellipsoid | Gaussian | 512 | 27159 | 2.11e+04–3.638e+04 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 3.931 |
| Rotated ill-conditioned ellipsoid | Gaussian | 1024 | 32361.2 | 2.959e+04–4.155e+04 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 3.908 |
| Rotated ill-conditioned ellipsoid | Gaussian | 2048 | 86437.3 | 7.731e+04–1.016e+05 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 3.839 |
| Rotated ill-conditioned ellipsoid | Gaussian | 5000 | 156474 | 1.404e+05–1.895e+05 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 3.918 |
| Rotated ill-conditioned ellipsoid | Local covariance | 128 | 20029.7 | 1.676e+04–2.366e+04 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 22.053 |
| Rotated ill-conditioned ellipsoid | Local covariance | 512 | 30606.4 | 2.152e+04–3.347e+04 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 8.650 |
| Rotated ill-conditioned ellipsoid | Local covariance | 1024 | 40543.9 | 3.666e+04–5.595e+04 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 6.524 |
| Rotated ill-conditioned ellipsoid | Local covariance | 2048 | 91365.1 | 7.425e+04–1.215e+05 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 5.338 |
| Rotated ill-conditioned ellipsoid | Local covariance | 5000 | 151488 | 1.431e+05–1.726e+05 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 4.719 |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 128 | 8010.1 | 4181–1.139e+04 | 0/10 | 0 | 99621–99753 | 707 | 128–128 | 1–1 | 6.880 |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 512 | 88487 | 5.324e+04–9.396e+04 | 0/10 | 0 | 98468–99016 | 174 | 512–512 | 1–1 | 7.208 |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 1024 | 129203 | 1.201e+05–1.663e+05 | 0/10 | 0 | 97797–98045 | 86 | 1024–1024 | 1–1 | 6.901 |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 2048 | 346085 | 2.366e+05–4.324e+05 | 0/10 | 0 | 94333–94551 | 41 | 2048–2048 | 1–1 | 7.368 |
| Rotated ill-conditioned ellipsoid | Bounded adaptive | 5000 | 784117 | 6.799e+05–9.093e+05 | 0/10 | 0 | 87367–87689 | 15 | 5000–5000 | 1–1 | 7.266 |
| Rotated ill-conditioned ellipsoid | Bounded + basin restarts | 128 | 8010.1 | 4181–1.139e+04 | 0/10 | 0 | 99621–99753 | 707 | 128–128 | 1–1 | 10.257 |
| Rotated ill-conditioned ellipsoid | Bounded + basin restarts | 512 | 88487 | 5.324e+04–9.396e+04 | 0/10 | 0 | 98468–99016 | 174 | 512–512 | 1–1 | 10.373 |
| Rotated ill-conditioned ellipsoid | Bounded + basin restarts | 1024 | 129203 | 1.201e+05–1.663e+05 | 0/10 | 0 | 97797–98045 | 86 | 1024–1024 | 1–1 | 10.255 |
| Rotated ill-conditioned ellipsoid | Bounded + basin restarts | 2048 | 346085 | 2.366e+05–4.324e+05 | 0/10 | 0 | 94333–94551 | 41 | 2048–2048 | 1–1 | 10.218 |
| Rotated ill-conditioned ellipsoid | Bounded + basin restarts | 5000 | 784117 | 6.799e+05–9.093e+05 | 0/10 | 0 | 87367–87689 | 15 | 5000–5000 | 1–1 | 10.656 |
| Rastrigin | Gaussian | 128 | 79.6846 | 77.11–85.82 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 2.120 |
| Rastrigin | Gaussian | 512 | 104.818 | 100.7–109.5 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 2.072 |
| Rastrigin | Gaussian | 1024 | 118.241 | 109.5–130.1 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 2.025 |
| Rastrigin | Gaussian | 2048 | 141.057 | 135.3–147.9 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 2.020 |
| Rastrigin | Gaussian | 5000 | 155.989 | 153.1–157.9 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 2.041 |
| Rastrigin | Local covariance | 128 | 83.1232 | 68.71–91.18 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 19.904 |
| Rastrigin | Local covariance | 512 | 100.27 | 96.89–108.3 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 6.901 |
| Rastrigin | Local covariance | 1024 | 119.708 | 111.6–122.7 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 4.551 |
| Rastrigin | Local covariance | 2048 | 137.229 | 127.4–143.1 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 3.419 |
| Rastrigin | Local covariance | 5000 | 168.306 | 157.9–172 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 2.771 |
| Rastrigin | Bounded adaptive | 128 | 95.5156 | 89.05–108.2 | 0/10 | 0 | 99618–99742 | 707 | 128–128 | 1–1 | 5.424 |
| Rastrigin | Bounded adaptive | 512 | 92.9896 | 87.25–95.07 | 0/10 | 0 | 98485–99029 | 174 | 512–512 | 1–1 | 5.344 |
| Rastrigin | Bounded adaptive | 1024 | 109.147 | 98.23–112 | 0/10 | 0 | 97797–98030 | 86 | 1024–1024 | 1–1 | 5.505 |
| Rastrigin | Bounded adaptive | 2048 | 133.165 | 128–140.6 | 0/10 | 0 | 94327–94563 | 41 | 2048–2048 | 1–1 | 5.459 |
| Rastrigin | Bounded adaptive | 5000 | 205.572 | 202.8–217 | 0/10 | 0 | 87369–87689 | 15 | 5000–5000 | 1–1 | 5.629 |
| Rastrigin | Bounded + basin restarts | 128 | 95.5156 | 89.05–108.2 | 0/10 | 0 | 99618–99742 | 707 | 128–128 | 1–1 | 7.841 |
| Rastrigin | Bounded + basin restarts | 512 | 92.9896 | 87.25–95.07 | 0/10 | 0 | 98485–99029 | 174 | 512–512 | 1–1 | 8.463 |
| Rastrigin | Bounded + basin restarts | 1024 | 109.147 | 98.23–112 | 0/10 | 0 | 97797–98030 | 86 | 1024–1024 | 1–1 | 8.181 |
| Rastrigin | Bounded + basin restarts | 2048 | 133.165 | 128–140.6 | 0/10 | 0 | 94327–94563 | 41 | 2048–2048 | 1–1 | 8.604 |
| Rastrigin | Bounded + basin restarts | 5000 | 205.572 | 202.8–217 | 0/10 | 0 | 87369–87689 | 15 | 5000–5000 | 1–1 | 8.719 |
| Rotated Rastrigin | Gaussian | 128 | 110.802 | 106.4–113 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 4.544 |
| Rotated Rastrigin | Gaussian | 512 | 130.461 | 123.3–138 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 4.563 |
| Rotated Rastrigin | Gaussian | 1024 | 146.05 | 141.2–154.4 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 4.471 |
| Rotated Rastrigin | Gaussian | 2048 | 203.156 | 199.9–208.7 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 4.483 |
| Rotated Rastrigin | Gaussian | 5000 | 266.773 | 261.8–310.4 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 4.496 |
| Rotated Rastrigin | Local covariance | 128 | 111.842 | 99.48–115 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 23.430 |
| Rotated Rastrigin | Local covariance | 512 | 131.189 | 121.8–138.1 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 9.300 |
| Rotated Rastrigin | Local covariance | 1024 | 147.855 | 140.6–154.8 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 7.048 |
| Rotated Rastrigin | Local covariance | 2048 | 201.245 | 194–204.3 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 5.677 |
| Rotated Rastrigin | Local covariance | 5000 | 291.788 | 259.1–318.8 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 5.365 |
| Rotated Rastrigin | Bounded adaptive | 128 | 360.697 | 316.6–394 | 0/10 | 0 | 99642–99751 | 707 | 128–128 | 1–1 | 7.629 |
| Rotated Rastrigin | Bounded adaptive | 512 | 263.962 | 222–309.9 | 0/10 | 0 | 98466–99015 | 174 | 512–512 | 1–1 | 7.296 |
| Rotated Rastrigin | Bounded adaptive | 1024 | 287.715 | 247.4–332.7 | 0/10 | 0 | 96937–97976 | 86 | 1024–1024 | 1–1 | 7.497 |
| Rotated Rastrigin | Bounded adaptive | 2048 | 365.699 | 348.3–375.9 | 0/10 | 0 | 94334–94556 | 41 | 2048–2048 | 1–1 | 7.568 |
| Rotated Rastrigin | Bounded adaptive | 5000 | 469.627 | 456.6–518.9 | 0/10 | 0 | 87367–87689 | 15 | 5000–5000 | 1–1 | 7.941 |
| Rotated Rastrigin | Bounded + basin restarts | 128 | 360.697 | 316.6–394 | 0/10 | 0 | 99642–99751 | 707 | 128–128 | 1–1 | 10.086 |
| Rotated Rastrigin | Bounded + basin restarts | 512 | 263.962 | 222–309.9 | 0/10 | 0 | 98466–99015 | 174 | 512–512 | 1–1 | 10.519 |
| Rotated Rastrigin | Bounded + basin restarts | 1024 | 287.715 | 247.4–332.7 | 0/10 | 0 | 96937–97976 | 86 | 1024–1024 | 1–1 | 11.176 |
| Rotated Rastrigin | Bounded + basin restarts | 2048 | 365.699 | 348.3–375.9 | 0/10 | 0 | 94334–94556 | 41 | 2048–2048 | 1–1 | 10.832 |
| Rotated Rastrigin | Bounded + basin restarts | 5000 | 469.627 | 456.6–518.9 | 0/10 | 0 | 87367–87689 | 15 | 5000–5000 | 1–1 | 10.826 |
| Rosenbrock | Gaussian | 128 | 45.7696 | 45.06–52.11 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 1.785 |
| Rosenbrock | Gaussian | 512 | 206.371 | 123–274 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 1.793 |
| Rosenbrock | Gaussian | 1024 | 1241.29 | 522.5–1807 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 1.707 |
| Rosenbrock | Gaussian | 2048 | 30854.2 | 1.558e+04–4.368e+04 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 1.688 |
| Rosenbrock | Gaussian | 5000 | 190967 | 1.43e+05–2.03e+05 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 1.764 |
| Rosenbrock | Local covariance | 128 | 33.5716 | 32.03–50.18 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 19.397 |
| Rosenbrock | Local covariance | 512 | 129.956 | 103.1–166.6 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 6.583 |
| Rosenbrock | Local covariance | 1024 | 669.246 | 436.2–910.5 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 4.289 |
| Rosenbrock | Local covariance | 2048 | 13767.2 | 1.004e+04–2.149e+04 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 3.174 |
| Rosenbrock | Local covariance | 5000 | 183064 | 1.557e+05–2.147e+05 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 2.517 |
| Rosenbrock | Bounded adaptive | 128 | 18.1796 | 17.77–18.88 | 0/10 | 0 | 99619–99736 | 707 | 128–128 | 1–1 | 4.988 |
| Rosenbrock | Bounded adaptive | 512 | 11557.5 | 8598–1.939e+04 | 0/10 | 0 | 98508–99031 | 175 | 512–512 | 1–1 | 5.042 |
| Rosenbrock | Bounded adaptive | 1024 | 168479 | 9.942e+04–2.042e+05 | 0/10 | 0 | 97766–98051 | 86 | 1024–1024 | 1–1 | 4.946 |
| Rosenbrock | Bounded adaptive | 2048 | 392231 | 3.542e+05–4.615e+05 | 0/10 | 0 | 94330–94555 | 41 | 2048–2048 | 1–1 | 5.023 |
| Rosenbrock | Bounded adaptive | 5000 | 595187 | 4.836e+05–6.891e+05 | 0/10 | 0 | 87368–87692 | 15 | 5000–5000 | 1–1 | 5.368 |
| Rosenbrock | Bounded + basin restarts | 128 | 18.1796 | 17.77–18.88 | 0/10 | 0 | 99619–99736 | 707 | 128–128 | 1–1 | 7.123 |
| Rosenbrock | Bounded + basin restarts | 512 | 11557.5 | 8598–1.939e+04 | 0/10 | 0 | 98508–99031 | 175 | 512–512 | 1–1 | 7.650 |
| Rosenbrock | Bounded + basin restarts | 1024 | 168479 | 9.942e+04–2.042e+05 | 0/10 | 0 | 97766–98051 | 86 | 1024–1024 | 1–1 | 7.696 |
| Rosenbrock | Bounded + basin restarts | 2048 | 392231 | 3.542e+05–4.615e+05 | 0/10 | 0 | 94330–94555 | 41 | 2048–2048 | 1–1 | 8.298 |
| Rosenbrock | Bounded + basin restarts | 5000 | 595187 | 4.836e+05–6.891e+05 | 0/10 | 0 | 87368–87692 | 15 | 5000–5000 | 1–1 | 8.225 |
| Boundary optimum (linear slope) | Gaussian | 128 | 35.9146 | 30.26–39.01 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 2.233 |
| Boundary optimum (linear slope) | Gaussian | 512 | 48.5692 | 38.73–53.36 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 2.135 |
| Boundary optimum (linear slope) | Gaussian | 1024 | 68.0491 | 59.18–73.64 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 2.128 |
| Boundary optimum (linear slope) | Gaussian | 2048 | 107.017 | 98.42–114.4 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 2.066 |
| Boundary optimum (linear slope) | Gaussian | 5000 | 142.077 | 133.9–151.5 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 2.099 |
| Boundary optimum (linear slope) | Local covariance | 128 | 34.7661 | 33.56–40.86 | 0/10 | 0 | 99968–99968 | 780 | 128–128 | 1–1 | 20.253 |
| Boundary optimum (linear slope) | Local covariance | 512 | 52.4476 | 40.99–64.07 | 0/10 | 0 | 99840–99840 | 194 | 512–512 | 1–1 | 7.108 |
| Boundary optimum (linear slope) | Local covariance | 1024 | 77.0552 | 63.64–79.62 | 0/10 | 0 | 99328–99328 | 96 | 1024–1024 | 1–1 | 4.639 |
| Boundary optimum (linear slope) | Local covariance | 2048 | 102.64 | 95.15–110.7 | 0/10 | 0 | 98304–98304 | 47 | 2048–2048 | 1–1 | 3.389 |
| Boundary optimum (linear slope) | Local covariance | 5000 | 137.722 | 133.6–142.8 | 0/10 | 0 | 100000–100000 | 19 | 5000–5000 | 1–1 | 2.849 |
| Boundary optimum (linear slope) | Bounded adaptive | 128 | 68.8694 | 53.22–79.71 | 0/10 | 0 | 99631–99740 | 707 | 128–128 | 1–1 | 4.769 |
| Boundary optimum (linear slope) | Bounded adaptive | 512 | 96.8033 | 83.95–102 | 0/10 | 0 | 98476–99019 | 174 | 512–512 | 1–1 | 5.115 |
| Boundary optimum (linear slope) | Bounded adaptive | 1024 | 135.86 | 131.1–140.5 | 0/10 | 0 | 97802–98031 | 86 | 1024–1024 | 1–1 | 5.498 |
| Boundary optimum (linear slope) | Bounded adaptive | 2048 | 170.259 | 164.8–182.9 | 0/10 | 0 | 94334–94553 | 41 | 2048–2048 | 1–1 | 5.418 |
| Boundary optimum (linear slope) | Bounded adaptive | 5000 | 186.309 | 183.8–191.2 | 0/10 | 0 | 87373–87682 | 15 | 5000–5000 | 1–1 | 5.643 |
| Boundary optimum (linear slope) | Bounded + basin restarts | 128 | 68.8694 | 53.22–79.71 | 0/10 | 0 | 99631–99740 | 707 | 128–128 | 1–1 | 6.314 |
| Boundary optimum (linear slope) | Bounded + basin restarts | 512 | 96.8033 | 83.95–102 | 0/10 | 0 | 98476–99019 | 174 | 512–512 | 1–1 | 8.029 |
| Boundary optimum (linear slope) | Bounded + basin restarts | 1024 | 135.86 | 131.1–140.5 | 0/10 | 0 | 97802–98031 | 86 | 1024–1024 | 1–1 | 8.168 |
| Boundary optimum (linear slope) | Bounded + basin restarts | 2048 | 170.259 | 164.8–182.9 | 0/10 | 0 | 94334–94553 | 41 | 2048–2048 | 1–1 | 8.545 |
| Boundary optimum (linear slope) | Bounded + basin restarts | 5000 | 186.309 | 183.8–191.2 | 0/10 | 0 | 87373–87682 | 15 | 5000–5000 | 1–1 | 8.353 |

## Reproduction

```sh
python3 fractal-gas-web/tools/benchmark-population-sweep.py --populations 128 512 1024 2048 5000 --seeds 10 --budget 100000 --workers 12 --output tests/optimization/reports/population-sweep-20d-100k
```

Use `--report-only` to regenerate tables from saved results. The referenced CMA data and its provenance are saved alongside the measured runs. Full requested/effective configurations and final diagnostics are in runs.jsonl.
