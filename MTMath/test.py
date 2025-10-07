import powerlaw
import joblib
import numpy as np

data = np.random.rand((100))
fit = powerlaw.Fit(data)
R, p = fit.distribution_compare("power_law", "lognormal")
joblib.dump(fit, "data.json")
