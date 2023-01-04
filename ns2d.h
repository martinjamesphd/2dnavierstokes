#ifndef ns2d_h
#define ns2d_h

extern rfftwnd_plan p_for, p_inv;
extern double *exp_dt;
extern fftw_complex *fomega_ft;
extern int sys_size, part_num, nthreads, ntheads_omp, tau_num; 
extern double dt, vis, mu,  tau_min, tau_max; 

struct coordinate
{
	double x;
	double y;
	double tau;
};
	
void init_den_state(int *den_state);

void init_omega(fftw_complex *omega_ft, int *den_state);

void find_vel_ft(fftw_complex *omega_ft, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft);

void find_e_spectra(fftw_complex *x_vel, fftw_complex *y_vel, double *e_spectra);

void gen_force(double famp, int kf);

void find_jacobian_ft(double *omega, double *x_vel, double *y_vel);

double find_energy(double *e_spectra);

double find_epsilon(fftw_complex *omega_ft);

void solve_rk2 (fftw_complex *omega_ft, fftw_complex *omega_t, fftw_complex *x_vel_ft, fftw_complex *y_vel_ft, struct coordinate *part_pos, struct coordinate *part_vel);

void init_part(struct coordinate *part_pos, struct coordinate *part_vel);

void update_part(double *x_vel, double *y_vel, struct coordinate *part_pos, struct coordinate *part_vel);

double linear_interp(double x_pos, double y_pos, double *vel);

#endif
