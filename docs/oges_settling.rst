OGES controller adapters
========================

``controllers/oges.py`` loads nominal dynamics, converts the state/reference
frames, and calls the Namor OGES controller API. The grasper uses its separate
PD command because it is outside the four-joint arm policy.

``controllers/dr_oges.py`` uses the same nominal call, consumes native observer
residuals, and calls Namor's DRCLFOGES projection API. Missing, stale, duplicate,
reset or model-incompatible observations cannot supply residual evidence.
With inactive residual evidence, the command equals nominal OGES.

The adapters contain no load adaptation, observer-tracking learning,
post-control filter, custom projection metric or separate diagnostics publisher.
Weights, effort limits and shared nominal-model selection remain explicit.
The controller algorithms live in Namor; these adapters do not reimplement them.
