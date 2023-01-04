#include<stdio.h>
#include<time.h>
#include<rfftw_threads.h>
#include<math.h>
#include"ns2d.h"

//function to intialize particle positions
void init_part(struct coordinate *part_pos, struct coordinate *part_vel)
{
	int i, j, part_i, index=part_num/tau_num;
	
	srand(time(NULL));	//seeds random number generator with current time
	
	//initializes particle positions randomly and velocity as 0
	for(j=0;j<tau_num;j++)
	for(i=0;i<index;i++)
	{
		part_i=j*index+i;
		part_pos[part_i].x=2.0*M_PI*rand()/RAND_MAX;
		part_pos[part_i].y=2.0*M_PI*rand()/RAND_MAX;
		part_pos[part_i].tau=tau_min+(double)j*(tau_max-tau_min)/tau_num;
		part_vel[part_i].x=0;
		part_vel[part_i].y=0;
	}

}

//function to update particle position and velocity
void update_part(double *x_vel_f, double *y_vel_f, struct coordinate *part_pos, struct coordinate *part_vel)
{
	double x_vel_inter, y_vel_inter, x_pos, y_pos, x_vel, y_vel;
		
	int i;
	
	//openmp_s
	# pragma omp parallel for schedule(static) private(i, x_vel_inter, y_vel_inter, x_pos, y_pos, x_vel, y_vel)
	for(i=0;i<part_num;i++)
	{
		x_pos = part_pos[i].x;
		y_pos = part_pos[i].y;
		x_vel = part_vel[i].x;
		y_vel = part_vel[i].y;
		
		//interpolates fluid velocity at particle positions
		x_vel_inter = linear_interp(x_pos, y_pos, x_vel_f);
		y_vel_inter = linear_interp(x_pos, y_pos, y_vel_f);
		
		//RK step 1
		x_pos+=dt*x_vel*0.5;
		y_pos+=dt*y_vel*0.5;
		
		x_vel=x_vel-dt*(x_vel-x_vel_inter)*0.5/part_pos[i].tau;
		y_vel=y_vel-dt*(y_vel-y_vel_inter)*0.5/part_pos[i].tau;
				
		//periodic boundary
		if(x_pos<0)
			x_pos+=2*M_PI;
		if(x_pos>=2*M_PI)
			x_pos-=2*M_PI;
		
		if(y_pos<0)
			y_pos+=2*M_PI;
		if(y_pos>=2*M_PI)
			y_pos-=2*M_PI;
			
		//interpolates fluid velocity at particle positions
		x_vel_inter = linear_interp(x_pos, y_pos, x_vel_f);
		y_vel_inter = linear_interp(x_pos, y_pos, y_vel_f);
		
		//RK step2
		part_pos[i].x+=dt*x_vel;
		part_pos[i].y+=dt*y_vel;
		
		part_vel[i].x-=dt*(x_vel-x_vel_inter)/part_pos[i].tau;
		part_vel[i].y-=dt*(y_vel-y_vel_inter)/part_pos[i].tau;
				
		//periodic boundary
		if(part_pos[i].x<0)
			part_pos[i].x+=2*M_PI;
		if(part_pos[i].x>=2*M_PI)
			part_pos[i].x-=2*M_PI;
		
		if(part_pos[i].y<0)
			part_pos[i].y+=2*M_PI;
		if(part_pos[i].y>=2*M_PI)
			part_pos[i].y-=2*M_PI;
	}
	//openmp_e
}

//function to interpolate fluid velocity at position (x_pos, y_pos) using bilinear interpolation technique
double linear_interp(double x_pos, double y_pos, double *vel)
{
	/*
	index_00, index_01, index_10, index_11 are indices of grid containing (x_pos,y_pos) as shown below
	
	
	index_01 _______________index_11
		|		|
		|x_pos,y_pos	|
	index_00|_______________|index_10
		
	*/
	
	int x_index, y_index, index_00, index_01, index_10, index_11;
	double vel_inter, vel_temp1, vel_temp2, x_index_val, y_index_val;
	
	x_index = x_pos*sys_size/(2*M_PI);	//scales x_pos and y_pos to sys_size
	y_index = y_pos*sys_size/(2*M_PI);
	
	x_index_val = x_index*(2*M_PI)/sys_size;
	y_index_val = y_index*(2*M_PI)/sys_size;
	
	index_00 = x_index*(sys_size+2) + y_index;	//finds indices
	index_01 = x_index*(sys_size+2) + y_index + 1;
	index_10 = (x_index+1)*(sys_size+2) + y_index;
	index_11 = (x_index+1)*(sys_size+2) + y_index +1;
	
	if(x_index==sys_size-1)		//accounts for edges
	{
		index_10 = y_index;
		index_11 = y_index+1;
	}
	
	if(y_index==sys_size-1)
	{
		index_01 = x_index*(sys_size+2);
		index_11 = (x_index+1)*(sys_size+2);
	}
	
	if(x_index==sys_size-1&&y_index==sys_size-1)
		index_11 = 0;
	
	//bilinear interpolation
	vel_temp1 = vel[index_00] + (vel[index_10]-vel[index_00])*(x_pos-x_index_val)/((2*M_PI)/sys_size);
	vel_temp2 = vel[index_01] + (vel[index_11]-vel[index_01])*(x_pos-x_index_val)/((2*M_PI)/sys_size);
	
	vel_inter = vel_temp1 + (vel_temp2-vel_temp1)*(y_pos-y_index_val)/((2*M_PI)/sys_size);
	
	return vel_inter;
}
