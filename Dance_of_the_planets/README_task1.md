# Dance of the planets

## Model explaination
- I used PINN model.
- Model is trained to minimise initial condition loss (ic loss) and ode loss.
- I first trained the model only on ic loss as it was quite large compared to ode loss.
- Model is then tarined to minimise ic + ode loss.

## Evaluation metrics
- initial conditions used: m1=1.0, m2=1.0 , x1=(-1,0), x2=(1,0), v1=(0,-0.5), v2=(0,0.5), t_pred=0.5
- When run on test dataset:
    - mean ode loss= 0.028.
    - MAE= 0.036467
- Predicted position: x1= (-1.015,-0.266), x2=(1.016, 0.266)