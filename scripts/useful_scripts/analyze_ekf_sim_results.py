import numpy as np 
import pickle

from numpy import array

from context import kmodel

from scipy import stats

with open('sim_rmse_results.pickle', 'rb') as handle:
    results = pickle.load(handle)

k_model, initial_conditions,subject_to_rmse_gf_dict,subject_to_rmse_avg_dict, ls_gf_list, gf_calibration, avg_calibration = results

# subject_to_rmse_gf_dict = {'AB01': [array([ 0.398,  0.497,  1.108, 12.128]), array([ 0.374,  0.712,  0.805, 21.369]), array([0.126, 0.054, 0.097, 2.42 ]), array([0.132, 0.089, 0.151, 3.348]), array([ 0.432,  0.695,  0.786, 19.673])], 'AB02': [array([0.266, 0.421, 0.538, 5.737]), array([0.116, 0.107, 0.132, 4.567]), array([0.103, 0.059, 0.085, 3.18 ]), array([0.103, 0.066, 0.07 , 3.951]), array([0.103, 0.065, 0.07 , 3.952])], 'AB03': [array([0.129, 0.076, 0.169, 3.308]), array([0.135, 0.104, 0.148, 4.145]), array([0.132, 0.099, 0.164, 3.072]), array([0.131, 0.069, 0.156, 3.379]), array([0.233, 0.349, 0.482, 4.155])], 'AB04': [array([0.154, 0.253, 0.301, 4.811]), array([ 0.309,  0.881,  0.969, 21.506]), array([0.138, 0.232, 0.286, 1.948]), array([0.389, 0.845, 0.997, 4.855]), array([ 0.411,  0.612,  0.694, 20.927])], 'AB05': [array([ 0.369,  0.726,  0.864, 21.   ]), array([ 0.453,  0.684,  1.045, 21.229]), array([0.116, 0.062, 0.092, 2.153]), array([0.118, 0.048, 0.091, 1.994]), array([ 0.404,  0.602,  0.926, 20.099])], 'AB06': [array([0.113, 0.066, 0.122, 1.908]), array([0.115, 0.069, 0.103, 3.217]), array([0.114, 0.054, 0.104, 2.051]), array([0.347, 0.46 , 0.788, 8.229]), array([ 0.405,  0.546,  0.988, 11.811])], 'AB07': [array([0.124, 0.064, 0.083, 2.119]), array([0.133, 0.106, 0.129, 4.369]), array([ 0.406,  0.686,  0.865, 13.531]), array([0.122, 0.054, 0.084, 2.176]), array([ 0.436,  0.703,  0.423, 20.96 ])], 'AB08': [array([0.114, 0.069, 0.142, 2.288]), array([0.115, 0.071, 0.163, 2.859]), array([ 0.439,  0.585,  0.909, 12.449]), array([ 0.32 ,  0.464,  0.639, 10.051]), array([0.124, 0.149, 0.218, 4.284])], 'AB09': [array([ 0.387,  0.511,  1.068, 11.53 ]), array([ 0.325,  0.706,  0.83 , 19.832]), array([ 0.322,  0.701,  0.825, 19.514]), array([0.132, 0.053, 0.164, 2.155]), array([0.131, 0.105, 0.152, 3.91 ])], 'AB10': [array([ 0.36 ,  0.504,  0.737, 12.13 ]), array([ 0.378,  0.819,  1.006, 20.542]), array([0.114, 0.053, 0.087, 2.935]), array([ 0.432,  0.704,  0.738, 14.44 ]), array([ 0.359,  0.589,  0.675, 15.567])]}
# subject_to_rmse_avg_dict = {'AB01': [array([0.123, 0.077, 0.107, 2.733]), array([0.123, 0.083, 0.115, 3.789]), array([0.12 , 0.059, 0.094, 2.693]), array([0.12 , 0.059, 0.094, 2.811]), array([0.12 , 0.066, 0.107, 2.913])], 'AB02': [array([0.112, 0.067, 0.081, 3.248]), array([0.119, 0.09 , 0.112, 3.675]), array([0.112, 0.079, 0.077, 3.248]), array([0.113, 0.067, 0.076, 3.492]), array([0.115, 0.078, 0.095, 3.418])], 'AB03': [array([0.135, 0.084, 0.185, 3.162]), array([0.136, 0.085, 0.182, 3.51 ]), array([0.134, 0.078, 0.184, 3.006]), array([0.134, 0.077, 0.18 , 3.15 ]), array([0.135, 0.085, 0.187, 3.357])], 'AB04': [array([0.178, 0.315, 0.388, 2.153]), array([0.184, 0.335, 0.414, 3.508]), array([0.177, 0.317, 0.391, 2.103]), array([0.179, 0.318, 0.395, 2.235]), array([0.177, 0.316, 0.391, 2.662])], 'AB05': [array([0.113, 0.059, 0.098, 2.416]), array([0.113, 0.06 , 0.09 , 2.842]), array([0.113, 0.069, 0.088, 2.213]), array([0.112, 0.048, 0.09 , 2.23 ]), array([0.116, 0.077, 0.123, 2.992])], 'AB06': [array([0.113, 0.067, 0.107, 2.222]), array([0.114, 0.069, 0.104, 2.698]), array([0.113, 0.069, 0.103, 2.066]), array([0.112, 0.059, 0.11 , 2.108]), array([0.113, 0.068, 0.105, 2.459])], 'AB07': [array([0.131, 0.07 , 0.096, 2.174]), array([0.131, 0.071, 0.088, 2.713]), array([0.13 , 0.062, 0.089, 1.99 ]), array([0.129, 0.06 , 0.089, 2.021]), array([0.135, 0.096, 0.121, 2.72 ])], 'AB08': [array([0.111, 0.079, 0.105, 2.185]), array([0.111, 0.08 , 0.097, 2.752]), array([0.111, 0.086, 0.096, 2.194]), array([0.111, 0.08 , 0.096, 2.522]), array([0.111, 0.078, 0.098, 2.409])], 'AB09': [array([0.128, 0.061, 0.109, 2.036]), array([0.132, 0.09 , 0.142, 3.252]), array([0.126, 0.056, 0.11 , 2.   ]), array([0.126, 0.053, 0.112, 2.043]), array([0.13 , 0.078, 0.129, 2.773])], 'AB10': [array([0.115, 0.073, 0.097, 3.147]), array([0.116, 0.075, 0.083, 3.491]), array([0.114, 0.061, 0.081, 3.056]), array([0.114, 0.065, 0.087, 3.115]), array([0.116, 0.074, 0.086, 3.361])]}

gf_rmse_list = []
avg_rmse_list = []
ls_gf_rmse_list = []


numpy_to_latex = lambda arr,ind: ' & '.join(["{:.4f}".format((arr[ind][i])) for i in range(3)])

for ls_gf, (subject_name_gf, value_gf),(subject_name_avg, value_avg) in zip(ls_gf_list,subject_to_rmse_gf_dict.items(),subject_to_rmse_avg_dict.items()):

    print(f'{subject_name_gf} & GF & ' + numpy_to_latex(value_gf,0) + ' & \\\\')
    print(f'{subject_name_avg} & AVG & ' + numpy_to_latex(value_avg,0) + ' & \\\\')
    print(f'{subject_name_avg} & LS GF & ' +  numpy_to_latex(value_avg,1) + ' & ' + str(ls_gf) + '\\\\')
    print('\\hline')


    if(subject_name_gf != "AB10"):
        gf_rmse_list.append(value_gf[0])
        avg_rmse_list.append(value_avg[0])
        ls_gf_rmse_list.append(value_avg[1])


gf_rmse_np = np.array(gf_rmse_list)
avg_rmse_np = np.array(avg_rmse_list)
ls_gf_rmse_np = np.array(ls_gf_rmse_list)

gf_rmse_np_mean = gf_rmse_np.mean(axis=0).reshape(1,-1)
avg_rmse_np_mean = avg_rmse_np.mean(axis=0).reshape(1,-1)
ls_gf_rmse_np_mean = ls_gf_rmse_np.mean(axis=0).reshape(1,-1)



print(f"Mean & GF & {numpy_to_latex(gf_rmse_np_mean,0)} \\\\")
print(f"Mean & AVG & {numpy_to_latex(avg_rmse_np_mean,0)} \\\\")
print(f"Mean & LS GF & {numpy_to_latex(ls_gf_rmse_np_mean,0)} \\\\")

print("\hline")

degrees_of_freedom = 10-1-1
t_value = (gf_rmse_np_mean - avg_rmse_np_mean)/(ls_gf_rmse_np - avg_rmse_np).std(axis=0)*np.sqrt(degrees_of_freedom)

# print(stats.ttest_rel(ls_gf_rmse_np,avg_rmse_np))
print(stats.t.sf(t_value, degrees_of_freedom))