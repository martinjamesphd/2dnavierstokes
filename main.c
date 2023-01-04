/*This program solves NS equations in 2 dimension
* for a periodic velocity using the pseudo-
* spectral algorithm*/

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<rfftw_threads.h>
#include"ns2d.h"

/* Following are the global variables.
  These variables do not change value
  during the execution of program*/
rfftwnd_plan p_for, p_inv;
double *exp_dt;			//variable to store time increments and aliasing (refer function init_den_state())
fftw_complex *fomega_ft;	//variable to store the external deterministic forcing term
int sys_size, part_num, nthreads, nthreads_omp, tau_num; 
double dt, vis, mu, tau_min, tau_max, rho_ratio; 

//starts function MAIN() 
int main(int argc, char *argv[])
{
	
	int max_iter, i, j, k, ij, jk, kf, k1, flag, pflag, s_flag;
	double famp, len_flow;
	double energy, epsilon, epsilon_val, epsilon_sum=0, tau_flow;
	FILE *fp, *fp_e;
	
	struct coordinate *part_pos, *part_vel;
	
	//create directories to store data
	system("mkdir -p spectra");
	system("mkdir -p field");
	system("mkdir -p particle");
	
	fp = fopen(argv[1],"r");
	
	//reads the variables from input file
	fscanf(fp,"%d%d%lE%lE%lE%lE%d%d%d%d%d%d", &sys_size, &max_iter, &dt, &vis, &famp, &mu, &kf, &nthreads, &nthreads_omp,\
	&flag, &pflag, &s_flag);
	
	fclose(fp);
	
	int y_size_f = sys_size/2+1;
	int y_size_r = sys_size+2;
	
	double scale = 1.0/(sys_size*sys_size);

	//Memory allocation
	double *omega 			= (double*)malloc(sys_size*(sys_size+2)*sizeof(double));
	fftw_complex *omega_ft		= (fftw_complex*)omega;
	fftw_complex *omega_t		= (fftw_complex*)malloc((sys_size*y_size_f)*sizeof(fftw_complex));
	double *x_vel 			= (double*)malloc(sys_size*(sys_size+2)*sizeof(double));
	fftw_complex *x_vel_ft		= (fftw_complex*)x_vel;		
	double *y_vel 			= (double*)malloc(sys_size*(sys_size+2)*sizeof(double));
	fftw_complex *y_vel_ft		= (fftw_complex*)y_vel;	
	double *e_spectra		= (double*)malloc(y_size_f*sizeof(double));
	int *den_state 			= (int*)malloc(y_size_f*sizeof(int));
	
	fomega_ft			= (fftw_complex*)malloc(sys_size*y_size_f*sizeof(fftw_complex));
	exp_dt				= (double*)malloc(sys_size*y_size_f*sizeof(double));

	//fftw variables for forward and inverse transform
	p_for = rfftw2d_create_plan(sys_size, sys_size, FFTW_REAL_TO_COMPLEX, FFTW_MEASURE | FFTW_IN_PLACE);
	p_inv = rfftw2d_create_plan(sys_size, sys_size, FFTW_COMPLEX_TO_REAL, FFTW_MEASURE | FFTW_IN_PLACE);
	
	double time=0.0;
	int t_flag=max_iter/s_flag;		//flags to store data
	char fname[35];				//stores filename
		
	fftw_threads_init();			//initializes threads for fftw

	if(argv[2][0]=='0')				//if no input omega file is given
	{						//initialize omega at t=0 by a periodic function
		init_den_state(den_state);
		init_omega(omega_ft, den_state);
	}
	
	else						//else read from the file
	{
		fp = fopen(argv[2],"r");
		for(i=0;i<sys_size;i++)
			for(j=0;j<y_size_f;j++)
			{
				ij = i*y_size_f+j;
				fscanf(fp,"%lf%lf",&omega_ft[ij].re,&omega_ft[ij].im);
			}
		init_den_state(den_state);
		fclose(fp);
	}

	fp = fopen(argv[3],"r");
	fscanf(fp,"%d%lf%lf%lf%d", &part_num, &tau_min, &tau_max, &rho_ratio, &tau_num);	//reads particle variables from file
	fclose(fp);
	
	//allocate memory for particle position and velocity variables
	part_pos = (struct coordinate*)malloc(part_num*sizeof(struct coordinate));
	part_vel = (struct coordinate*)malloc(part_num*sizeof(struct coordinate));


	if(argv[4][0]=='0')	//if no input particle data is given initializes by a random function
		init_part(part_pos, part_vel);
	
	else			//else read from file
	{
		fp=fopen(argv[4],"r");
		for(j=0;j<part_num;j++)
			fscanf(fp,"%lf%lf%lf%lf%lf\n", &part_pos[j].tau, &part_pos[j].x, &part_pos[j].y\
			, &part_vel[j].x, &part_vel[j].y);
		fclose(fp);
	}
	
	gen_force(famp, kf);		//generates stirring force
	
	rfftwnd_threads_one_complex_to_real(nthreads, p_inv, omega_ft, NULL);	//inverse fourier transforms and fourier transforms
	for(j=0;j<sys_size;j++)							//the initialized omega
		for(k=0;k<y_size_r;k++)
		{	
			jk=j*y_size_r+k;
			omega[jk]*=scale;
		}
	rfftwnd_threads_one_real_to_complex(nthreads, p_for, omega, NULL);
	find_vel_ft( omega_ft, x_vel_ft, y_vel_ft);
	
	fp_e=fopen("spectra/energy.dat","w");	//file to store energy vs time data
	fprintf(fp_e,"#time-tot_Energy-epsilon_inst-epsilon_avg-tau_flow-len_flow\n");
	
	//time marching
	for(k=0;k<s_flag;k++)
	{
		if(pflag==0)	//if pflag is set zero, store the vorticity profile
		{
			rfftwnd_threads_one_complex_to_real(nthreads, p_inv, omega_ft, NULL);
			for(j=0;j<sys_size;j++)						
				for(k1=0;k1<y_size_r;k1++)
				{	
					jk=j*y_size_r+k1;
					omega[jk]*=scale;
				}
			
			sprintf(fname,"field/velProfile%.3d.dat",k);		
			fp=fopen(fname,"w");
			fprintf(fp,"#Vorticity profile at time %f\n#W\n",time);		
			for(j=0;j<sys_size;j=j+flag)	//writes data to file
			{
				for(k1=0;k1<y_size_r;k1=k1+flag)
				{
					jk=j*y_size_r+k1;
					fprintf(fp,"%f %f %E\n",(double)j*M_PI*2/sys_size, \
						(double)k1*M_PI*2/sys_size, omega[jk]);
				}
				fprintf(fp,"\n");
			}
			fclose(fp);
			rfftwnd_threads_one_real_to_complex(nthreads, p_for, omega, NULL);
		}
		
		epsilon_val=find_epsilon(omega_ft);	//calculates energy dissipation rate
		epsilon_sum+=epsilon_val;
		epsilon=epsilon_sum/(k+1);
		tau_flow=sqrt(vis/epsilon);		//Kolmogorov time scale and length scale
		len_flow=sqrt(sqrt(vis*vis*vis/epsilon));
		find_e_spectra(x_vel_ft, y_vel_ft, e_spectra);	//calculates energy spectra
		sprintf(fname,"spectra/eSpectra%.3d.dat",k);		
		fp=fopen(fname,"w");
		fprintf(fp,"#Energy spectra at time %f\n",time);
		for(j=1;j<y_size_f;j++)
			fprintf(fp, "%d %E\n", j, e_spectra[j]);	//stores the energy spectra
		fclose(fp);
	
		energy=find_energy(e_spectra);		//calculate total energy
		fprintf(fp_e,"%f %E %E %E %f %E\n", time, energy, epsilon_val, epsilon, tau_flow, len_flow);
		fflush(fp_e);

		fp=fopen("init_flow.dat","w");		//stores current vorticity profile (in fourier space)
		for(i=0;i<sys_size;i++)			//for future run 
			for(j=0;j<y_size_f;j++)
			{
				ij=i*y_size_f+j;
				fprintf(fp, "%E %E ", omega_ft[ij].re, omega_ft[ij].im);
			}
		fclose(fp);
		
		sprintf(fname,"particle/particle%.3d.dat",k);
		fp=fopen(fname,"w");
		fprintf(fp, "#Particle position at time %f\n", time);	//stores particle position
		for(j=0;j<part_num;j++)
			fprintf(fp,"%f %f %f %f %f\n", part_pos[j].tau/tau_flow, part_pos[j].x,part_pos[j].y,\
			 part_vel[j].x, part_vel[j].y);
		fclose(fp);
		
		fp=fopen("init_part.dat","w");
		for(j=0;j<part_num;j++)
			fprintf(fp,"%f %f %f %f %f\n",part_pos[j].tau, part_pos[j].x,part_pos[j].y\
			, part_vel[j].x, part_vel[j].y);
		fclose(fp);
				
		for(i=0;i<t_flag;i++)	
		{	
			/*Solves the differential in fourier space using Runka Kutta 2
			takes current omega_ft, x_velocity_ft and y_velocity_ft and gives the updated omega_ft
			omega_t is a temporary variable. Note that current x_vel_ft and y_vel_ft gets destroyed*/
			solve_rk2(omega_ft, omega_t, x_vel_ft, y_vel_ft, part_pos, part_vel);    
			find_vel_ft( omega_ft, x_vel_ft, y_vel_ft);
		}
		time=time+dt*t_flag;	
	}
	fclose(fp_e);		

	fp=fopen("init_flow.dat","w");		//stores current vorticity profile (in fourier space)
	for(i=0;i<sys_size;i++)			//for future runs
		for(j=0;j<y_size_f;j++)
		{
			ij=i*y_size_f+j;
			fprintf(fp, "%E %E ", omega_ft[ij].re, omega_ft[ij].im);
		}
	fclose(fp);
	
	fp=fopen("init_part.dat","w");
	for(j=0;j<part_num;j++)
		fprintf(fp,"%f %f %f %f %f\n",part_pos[j].tau, part_pos[j].x,part_pos[j].y\
		, part_vel[j].x, part_vel[j].y);
	fclose(fp);
		
	rfftwnd_destroy_plan(p_for);	//free heap variables
	rfftwnd_destroy_plan(p_inv);

	free(omega);
	free(omega_t);
	free(x_vel);
	free(y_vel);
	free(exp_dt);
	free(fomega_ft);
	free(den_state);
	free(e_spectra);
	free(part_pos);
	free(part_vel);
	
	return 0;
}//end of MAIN()
