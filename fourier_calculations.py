import numpy as np

#My implementation based on Grays least squares estimation
def get_fourier_coeff (x,y,z,n=0):
    #flatten all the data out
    #phi=x.reshape(-1)
    phi=2*np.pi*x.reshape(-1)
    step_length=y.reshape(-1)
    left_hip_angle=z.reshape(-1)
 
    R = [np.ones((len(phi),))]
    for i in range(n)[1:]:
        R.append(np.sin(i*phi))
        R.append(np.cos(i*phi))
        R.append(step_length*np.cos(i*phi))
        R.append(step_length*np.sin(i*phi))
    R = np.array(R).T
        
    return np.linalg.solve(R.T @ R, R.T @ left_hip_angle)
 
    

def get_fourier_sum(a,x,y,n):
    phi=2*np.pi*x.reshape(-1)
    step_length=y.reshape(-1)
 
    R = [np.ones((len(phi),))]
    for i in range(n)[1:]:
        R.append(np.sin(i*phi))
        R.append(np.cos(i*phi))
        R.append(step_length*np.cos(i*phi))
        R.append(step_length*np.sin(i*phi))
    R = np.array(R).T
    
    return R @ a




#Todo: remove the phi_total[0] to just phi
def get_fourier_prediction(fourier_coeff, phi_total, step_length_total, num_params):
    return np.array([get_fourier_sum(fourier_coeff, phi, step, num_params)\
              for phi,step in zip(phi_total, step_length_total)]).reshape(-1)

