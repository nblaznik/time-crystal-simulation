// Simulation of Schrodinger equation.
// Including inverse time evolution to find the ground state.


// Compile using
// gcc main_forlei.c -o gpe_sim -lm -lfftw3 -lfftw3_threads -fopenmp

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <omp.h>
#include <time.h>
#include <fftw3.h>

// Constants etc.

#define Pi 3.141592653589793 // Pi
#define h 6.62606896E-34 // Plancks constant
#define hbar 1.054571628E-34 // Plancks reduced constant
#define kB 1.3806504E-23 // Boltzmann constant
#define mass 3.8175405E-26 // Sodium mass
#define eps0 8.8541878176E-12 // vaccuum permittivity
#define me 9.10938215E-31 // Electron mass
#define qe 1.602176487E-19 // Electron charge
#define as11 2.80358086E-9 // s-wave scatt length
#define as12 2.80358086E-9 // s-wave scatt length
#define as22 2.704248E-9 // s-wave scatt length

// Temporary definition of MAX function
#define MAX(a,b) (((a)>(b))?(a):(b))

// System structure definition
struct dir {
	double w;
	double losc;		// Oscillator length
	double tosc;		// Oscillator period
	double s;			// System size
	double * x;		// System grid
	double dx;			// Lattice constant
	double * k;		// reciprocal system grid
	double dk;			// reciprocal lattice constant
	double rtf;		// Thomas Fermi radius
	double rtfs;		// Scaled Thomas Fermi radius
	int gridsize;
};

struct idir {
	double w;
	double losc;		// Oscillator length
};

struct system {
	struct dir x;
	struct dir y;
	struct idir z;			// In 2D, this is the parameter that is integrated out!
	double lscale;		// Length scaling
	double tscale;			// Time scaling
	double dt;				// Size of time step
	double * k2;			//  k-squared array
	double * x2;			// x-squared array
	double complex consx;
	double complex consk;
	double complex consc;	// Centrifugal term
	double complex consint11;
	double complex consint12;
	double complex consint22;
	double complex consmu;
	fftw_complex * psi1;		// Wave function 1
	fftw_complex * psi2;		// Wave function 2
	fftw_plan pf1;			// Plan forward
	fftw_plan pb1;			// Plan back
	fftw_plan pf2;			// Plan forward
	fftw_plan pb2;			// Plan back
	int gridsize;			// Grid size
	double npart;			// Number of particles in the system
	double mu;			// Chemical potential of the system
	double mus;			// Scaled (Dimensionless) chemical potential of the system
	double gs11;			// Scaled U0
	double gs12;			// Scaled U0
	double gs22;			// Scaled U0
	double * extrapot;		// Added potential
	double * field1t;
	double * field2t;
	double complex * consfield11;
	double complex * consfield12;
	double complex * consfield21;
	double complex * consfield22;
	double dampfactor;
	double ramandet;
	double moddepth;		// Radial quadrupole modulation depth.
	int startpoint;
	int currpoint;
	int endpoint;
};

struct matr {
	double complex matrix[4];
	double complex eigenvalue[2];
	double complex eigenvector[4];

};

// Function definitions
struct system normalize(struct system sys);
struct system initextrapot(struct system sys);
double densintegrate(struct system sys);
double enerintegrate(struct system sys);
double densintegrate1(struct system sys);
double densintegrate2(struct system sys);
double energy1(struct system sys);
struct system initialize();
struct system timestep(struct system sys);
struct system imaginarytimestep(struct system sys);
double innerproductwithgroundstate(struct system sys);
void writetofile(struct system sys,char filename[]);
struct system excite(struct system sys); // Function was removed
struct system readfile(char filename[]);

// Matrix operation functions
void eigenval(struct matr * mat);
void eigenvec(struct matr * mat);
void expmatr(struct matr * mat);
void expmatrwr(struct matr * mat);


// Main loop
void main(int argc, char * argv[]){
	int ii; // Initialize iterators
	char filename[50];
	
	// TO CHANGE FILE NAME, CHANGE THIS PARAMETER
	char fileprefix[] = "rawdata/rad1kick_20p_7E-4damp";

	printf("Program %s started.\n",argv[0]);
	fflush(stdout);

	// Initialize the multithreading. (Not enabled on students server)
	omp_set_num_threads(30);
	fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());

	// Initialize the system structure,
	struct system sys;

	if(argc > 1){
		printf("Called with argument \"%s\". Attempting to open file.\n",argv[1]);
		fflush(stdout);
		sys = readfile(argv[1]);
	} else {
		printf("Called with no arguments. Will initialize system with standard parameters.\n");
		fflush(stdout);
		sys = initialize();
	}

	// Initialize extra potential
	// sys = initextrapot(sys);


	if(sys.currpoint<1){
		printf("Starting imaginary time evolution.\n");
		fflush(stdout);
	}

	while(sys.currpoint<1){
		if(sys.currpoint % 100 == 0){
			printf("Step %d;\t psi.dot.groundstate =  %E\n",sys.currpoint,innerproductwithgroundstate(sys));
			fflush(stdout);
			sprintf(filename, "%s%08d.dat",fileprefix,sys.currpoint);
			writetofile(sys,filename);
			printf("%d,%.16E,%.5E\n",sys.currpoint,energy1(sys),densintegrate1(sys));
			fflush(stdout);
		}
		sys = imaginarytimestep(sys);
	}
	printf("Done with imaginary time evolution!\n");
	fflush(stdout);

	sprintf(filename, "%s_gs.dat",fileprefix);
	writetofile(sys,filename);



	// Excite system.
	//sys = excite(sys);

	sys = normalize(sys);
	
	printf("Starting time evolution.\n");
	fflush(stdout);

	
	while(sys.currpoint<sys.endpoint){
		if(sys.currpoint % 100 == 0){
			printf("Step %d/%d\tEnergy: %E\tNorm: %E\n",sys.currpoint,sys.endpoint,energy1(sys),densintegrate1(sys));
			fflush(stdout);
			sprintf(filename, "%s%08d.dat",fileprefix,sys.currpoint);
			writetofile(sys,filename);
		}
		//sys.currpoint++;
		sys = timestep(sys);
	}

	printf("Done with time evolution! Wrapping up.\n");
	fflush(stdout);

	sprintf(filename, "%s_done.dat",fileprefix);
	writetofile(sys,filename);

	return;
}


// This will intialize the system construct for use
struct system initialize(){
	int ii,jj; // Local loop iterator

	struct system sys; // Define the system construct here.

	// Define the system sizes and timesteps
	sys.x.gridsize = 1024;
	sys.y.gridsize = 256;
	sys.gridsize = sys.x.gridsize*sys.y.gridsize;
	sys.dt = 1./1000.;
	sys.dampfactor = 7E-4;	// TO CHANGE DAMPING. CHANGE THIS PARAMETER.
	sys.ramandet = 250E6; // 250 MHz
	sys.moddepth = 0.2; // TO CHANGE MODULATION DEPTH, CHANGE THIS PARAMETER

	// Negative for imaginary time evolution
	sys.startpoint = -5000;
	sys.endpoint = 100001;
	sys.currpoint = sys.startpoint;

	// Allocate memory
	sys.psi1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sys.gridsize);
	sys.psi2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sys.gridsize);
	sys.k2 = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.x2 = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.extrapot = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.consfield11 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.consfield12 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.consfield21 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.consfield22 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.field1t = (double*) malloc(sizeof(double) * sys.endpoint);
	sys.field2t = (double*) malloc(sizeof(double) * sys.endpoint);
	sys.x.x = (double*) malloc(sizeof(double) * sys.x.gridsize);
	sys.x.k = (double*) malloc(sizeof(double) * sys.x.gridsize);
	sys.y.x = (double*) malloc(sizeof(double) * sys.y.gridsize);
	sys.y.k = (double*) malloc(sizeof(double) * sys.y.gridsize);

	if((sys.psi1==NULL)||(sys.psi2==NULL)||(sys.k2==NULL)||(sys.x2==NULL)||(sys.extrapot==NULL)||(sys.x.x==NULL)||(sys.x.k==NULL)||(sys.y.x==NULL)||(sys.y.k==NULL))
		printf("Memory wrongly allocated!\n");

	// Initialize values for the x-direction
	sys.x.w = 2.*Pi* 1.39;
	sys.x.losc = sqrt(hbar/(mass*sys.x.w));
	sys.x.tosc = 2.*Pi/sys.x.w;

	// Initialize values for the y-direction
	sys.y.w = 2.*Pi* 28.42;
	sys.y.losc = sqrt(hbar/(mass*sys.y.w));
	sys.y.tosc = 2.*Pi/sys.y.w;

	// Include information about the direction that was integrated out
	sys.z.w = 2.*Pi * 28.42;
	sys.z.losc = sqrt(hbar/(mass*sys.z.w));

	// Define particle number and calculate mu (calculate from mu later?)
	sys.npart = 10E7; // 0.0002 compensates for tighter trap (2000Hz)
	sys.mu = 1.47708846953*pow(sys.npart*as11/pow(sys.x.losc*sys.y.losc*sys.z.losc,1./3.),2./5.)*hbar*pow(sys.x.w*sys.y.w*sys.z.w,1./3.); // prefactor = 15^(2/5)/2

	// Calculate Thomas-Fermi radii
	sys.x.rtf = sqrt(2.*sys.mu/(mass*sys.x.w*sys.x.w));
	sys.y.rtf = sqrt(2.*sys.mu/(mass*sys.y.w*sys.y.w));

	// Initialize scaled parameters
	sys.tscale = sys.y.tosc;
	sys.lscale = sys.y.losc;	
	sys.mus = sys.mu*sys.tscale/hbar; // Calculate the scaled chemical potential
	sys.gs11 = 4*Pi*hbar*as11*sys.tscale/(mass * pow(sys.lscale,3.));
	sys.gs12 = 4*Pi*hbar*as12*sys.tscale/(mass * pow(sys.lscale,3.));
	sys.gs22 = 4*Pi*hbar*as22*sys.tscale/(mass * pow(sys.lscale,3.));

	sys.x.rtfs = sys.x.rtf/sys.lscale;
	sys.x.s = 2.* sys.x.rtfs;		// System size is from -param*lscale to param*lscale
	sys.x.dx = 2.*sys.x.s /(double)sys.x.gridsize;
	sys.x.dk = Pi/sys.x.s;

	sys.y.rtfs = sys.y.rtf/sys.lscale;
	sys.y.s = 2.* sys.y.rtfs;		// System size is from -param*lscale to param*lscale
	sys.y.dx = 2.*sys.y.s /(double)sys.y.gridsize; // Symmetric
	//sys.y.dx = sys.y.s /(double)sys.y.gridsize;	// Assymmetric
	sys.y.dk = Pi/sys.y.s;

	// Initialize the grids
	for(ii=0;ii<sys.x.gridsize;ii++){
		sys.x.x[ii] = -sys.x.s + sys.x.dx*((double)ii+0.5);
		if(ii<sys.x.gridsize/2)
			sys.x.k[ii] = sys.x.dk*(double)ii;
		else
			sys.x.k[ii] = sys.x.dk*(double)(sys.x.gridsize-ii);
	}

	for(ii=0;ii<sys.y.gridsize;ii++){
		sys.y.x[ii] = -sys.y.s + sys.y.dx*((double)ii+0.5); // Symmetric
		//sys.y.x[ii] = sys.y.dx*((double)ii+0.5); // Assymmetric
		if(ii<sys.y.gridsize/2)
			sys.y.k[ii] = sys.y.dk*(double)ii;
		else
			sys.y.k[ii] = sys.y.dk*(double)(sys.y.gridsize-ii);
	}

	// Initialize the k-sq grid
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			sys.k2[ii*sys.y.gridsize+jj] = sys.x.k[ii]*sys.x.k[ii]+sys.y.k[jj]*sys.y.k[jj];
			sys.x2[ii*sys.y.gridsize+jj] = (sys.x.w/sys.y.w)*(sys.x.w/sys.y.w)*sys.x.x[ii]*sys.x.x[ii]+sys.y.x[jj]*sys.y.x[jj];
		}
	}

	// Initialize fftw plans
	sys.pf1 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi1, sys.psi1, FFTW_FORWARD, FFTW_MEASURE);
	sys.pb1 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi1, sys.psi1, FFTW_BACKWARD, FFTW_MEASURE);
	sys.pf2 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi2, sys.psi2, FFTW_FORWARD, FFTW_MEASURE);
	sys.pb2 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi2, sys.psi2, FFTW_BACKWARD, FFTW_MEASURE);

	// Initialize the wavefunction in the Thomas Fermi profile
	//double tempconst = sys.lscale*sqrt(pow(sys.mu,3./2.)/(3.*sqrt(3.*Pi)*pow(hbar*sys.z.w,3./2.)*as11*sys.z.losc));//*sys.npart));
	double tempconst = sqrt(Pi*sys.mus/sys.gs11);
	fflush(stdout);
	//#pragma omp parallel for
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			sys.psi1[ii*sys.y.gridsize+jj] = sqrt(fabs(sys.y.x[jj]))*tempconst*pow(MAX(0.,1-(sys.x.x[ii]/sys.x.rtfs)*(sys.x.x[ii]/sys.x.rtfs)-(sys.y.x[jj]/sys.y.rtfs)*(sys.y.x[jj]/sys.y.rtfs)+hbar/(mass * sys.y.w * sys.lscale * sys.lscale) * sys.y.w * sys.tscale/(sys.mus*8.*sys.y.x[jj]*sys.y.x[jj])),1./2.);
			sys.psi2[ii*sys.y.gridsize+jj] = 0;
		}
	}

	// Initialize the simulation constants
	sys.consx = -0.25*(1.*I+sys.dampfactor)*(mass * sys.y.w * sys.lscale * sys.lscale)/hbar * sys.y.w * sys.tscale * sys.dt; // Divided by 2 because we do 2 seperate space steps
	sys.consk = -0.5*(1.*I+sys.dampfactor)*hbar/(mass * sys.y.w * sys.lscale * sys.lscale) * sys.y.w * sys.tscale * sys.dt;
	sys.consc = -0.125*sys.consk; // (/4 from nabla, and another /2 because it is in the potential (thus computed twice))
	sys.consmu = 0.5*(1.*I+sys.dampfactor)*sys.mus*sys.dt;
	//sys.consint11 = -0.5*0.75*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*sys.gs11/(pow(Pi,2./3.)*pow(sys.z.losc,4./3.)*pow(as11,1./3.))*sys.dt; // Divided by 2 because we do 2 seperate space steps

	// Interaction constants
	//double gavg = (sys.gs11+2.*sys.gs12+sys.gs22)/4.;
	//double asavg = (as11+2.*as12+as22)/4.;
	
	sys.consint11 = -0.5*(1.*I+sys.dampfactor)*(sys.gs11/Pi)*sys.npart*sys.dt;
	sys.consint12 = 0.;//-0.5*(1.*I+sys.dampfactor)*0.25*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*(2*sys.gs12+gavg)/(pow(Pi,2./3.)*pow(sys.z.losc,4./3.)*pow(asavg,1./3.))*sys.dt;
	sys.consint22 = 0.;//-0.5*(1.*I+sys.dampfactor)*0.25*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*(2*sys.gs22+gavg)/(pow(Pi,2./3.)*pow(sys.z.losc,4./3.)*pow(asavg,1./3.))*sys.dt;

	// Coupling arrays
	/*double raman1 = 1.5E-29 * 1./hbar; //d=1.5E-29 Cm * 10 V/m /hbar
	double raman2 = 1.5E-29 * 1./hbar;

	double complex coup11 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w/hbar * raman1*conj(raman1)/sys.ramandet; // dt* coefficient scaling * 
	double complex coup12 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w/hbar * raman2*conj(raman1)/sys.ramandet;
	double complex coup21 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w/hbar * raman2*conj(raman1)/sys.ramandet;
	double complex coup22 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w/hbar * raman2*conj(raman2)/sys.ramandet;*/

	// Arrays, since we want to add profile to these.
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			sys.consfield11[ii*sys.y.gridsize+jj] = 0;//coup11;
			sys.consfield21[ii*sys.y.gridsize+jj] = 0;//coup21;
			sys.consfield12[ii*sys.y.gridsize+jj] = 0;//coup12;
			sys.consfield22[ii*sys.y.gridsize+jj] = 0;//coup22;
		}
	}

	// Adding time-dependence
	for(ii=0;ii<sys.endpoint;ii++){
		
		// TO CHANGE TIME DEPENDENCE. MODIFY BELOW.
		// TIMES IN SECONDS
		
		if(((double)ii)*sys.tscale*sys.dt < 20E-3)
		{
			sys.field1t[ii] = sin((2*Pi/(20E-3)) * (double)ii *sys.tscale*sys.dt);
		}
		else
		{
			sys.field1t[ii] = 0;
		}
		sys.field2t[ii] = 0;
		
	}
	
	printf("mu: %E,%E\n",sys.mu/(2*Pi*hbar),sys.mus);
	printf("consint11: %E,%E\n",creal(sys.consint11),cimag(sys.consint11));
	printf("consint22: %E,%E\n",creal(sys.consint22),cimag(sys.consint22));
	printf("consint12: %E,%E\n",creal(sys.consint12),cimag(sys.consint12));
	//printf("coup11: %E,%E\n",creal(coup11),cimag(coup11));
	//printf("2Pi*RamanFreq: %E\n",raman1*raman2/sys.ramandet);


	//printf("List of constants:\nmu: %E\nnpart: %E\nx: %E\nk: %E\nint: %E\ncohl: %E\ndx: %E\ndy: %E\n",sys.mu,sys.npart,sys.consx,sys.consk,sys.consint11,(hbar/sqrt(2.*mass*sys.mu))/sys.lscale,sys.x.dx,sys.y.dx);

	return normalize(sys);
}

// Initialize the extra potential.
struct system initextrapot(struct system sys){
	int ii,jj;
	//double gausstemp;
	//gausstemp = sys.tscale*2*Pi*2000./(sqrt(2*Pi)*10.*sys.x.dx)*sys.dt;
	
	double modfactor = 0.2;

	#pragma omp parallel for
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			sys.extrapot[ii*sys.y.gridsize+jj] = modfactor*2.*sys.y.x[jj]*sys.y.x[jj];
		}
	}

	printf("Initialized extra potential.\n");

	return sys;
}

struct system excite(struct system sys){
	int ii,jj;
	double complex phase1,phase2;

	/*for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			
			phase1 = ((sys.y.x[jj]-sys.y.x[sys.y.gridsize/2-50])*I+sys.x.x[ii])/cabs((sys.y.x[jj]-sys.y.x[sys.y.gridsize/2-50])*I+sys.x.x[ii]);
			phase2 = ((sys.y.x[jj]-sys.y.x[sys.y.gridsize/2+50])*I-sys.x.x[ii])/cabs((sys.y.x[jj]-sys.y.x[sys.y.gridsize/2+50])*I-sys.x.x[ii]);
		//	phase = (sys.x.x[ii]+sys.y.x[jj]*I)/ffieldcabs(sys.x.x[ii]+sys.y.x[jj]*I);
			if(isfinite(phase1)==0)
				phase1 = 0;

			if(isfinite(phase2)==0)
				phase2 = 0;
	
			sys.psi2[ii*sys.y.gridsize+jj] = phase1*sys.psi1[ii*sys.y.gridsize+jj]/sqrt(2.);
			sys.psi1[ii*sys.y.gridsize+jj] = phase2*sys.psi1[ii*sys.y.gridsize+jj]/sqrt(2.);//phase;
		}
	}*/


	return sys;
}


void writetofile(struct system sys,char filename[]){
	int ii,jj; // Initialize iterators.

	FILE * file = fopen(filename,"w");

	fprintf(file,"%d\t%d\t%d\t%d\n",sys.currpoint,sys.endpoint,sys.x.gridsize,sys.y.gridsize);
	fprintf(file,"%E\t%E\t%E\t%E\t%E\n",sys.tscale,sys.dt,sys.lscale,sys.npart,sys.dampfactor);
	fprintf(file,"%E\t%E\t%E\n",sys.x.w,sys.y.w,sys.z.w);
	fprintf(file,"%E\t%E\n",sys.x.s,sys.y.s);
	
	for(ii=0;ii<sys.gridsize;ii++){
		fprintf(file,"%E\t%E\n",creal(sys.psi1[ii]),cimag(sys.psi1[ii]));
	}
	///if(sys.currpoint>0){
	//	for(ii=0;ii<sys.gridsize;ii++){
	//		fprintf(file,"%E\t%E\n",creal(sys.psi2[ii]),cimag(sys.psi2[ii]));
	//	}
	//}
	fclose(file);
	return;
}

struct system readfile(char filename[]){
	int ii,jj; // Initialize iterators.

	struct system sys;
	FILE * file = fopen(filename,"r");

	fscanf(file,"%d\t%d\t%d\t%d\n",&sys.currpoint,&sys.endpoint,&sys.x.gridsize,&sys.y.gridsize);
	fscanf(file,"%lE\t%lE\t%lE\t%lE\t%lE\n",&sys.tscale,&sys.dt,&sys.lscale,&sys.npart,&sys.dampfactor);
	fscanf(file,"%lE\t%lE\t%lE\n",&sys.x.w,&sys.y.w,&sys.z.w);
	fscanf(file,"%lE\t%lE\n",&sys.x.s,&sys.y.s);

	printf("%d\t%d\t%d\t%d\n%E\t%E\t%E\t%E\t%E\n%E\t%E\t%E\n%E\t%E\n",sys.currpoint,sys.endpoint,sys.x.gridsize,sys.y.gridsize,sys.tscale,sys.dt,sys.lscale,sys.npart,sys.dampfactor,sys.x.w,sys.y.w,sys.z.w,sys.x.s,sys.y.s);

	sys.gridsize = sys.x.gridsize*sys.y.gridsize;

	// Allocate memory (before import!)
	sys.psi1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sys.gridsize);
	sys.psi2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * sys.gridsize);
	sys.k2 = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.x2 = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.extrapot = (double*) malloc(sizeof(double) * sys.gridsize);
	sys.consfield11 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.consfield12 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.consfield21 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.consfield22 = (double complex*) malloc(sizeof(double complex) * sys.gridsize);
	sys.field1t = (double*) malloc(sizeof(double) * sys.endpoint);
	sys.field2t = (double*) malloc(sizeof(double) * sys.endpoint);
	sys.x.x = (double*) malloc(sizeof(double) * sys.x.gridsize);
	sys.x.k = (double*) malloc(sizeof(double) * sys.x.gridsize);
	sys.y.x = (double*) malloc(sizeof(double) * sys.y.gridsize);
	sys.y.k = (double*) malloc(sizeof(double) * sys.y.gridsize);

	// Initialize fftw plans (before import!)
	sys.pf1 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi1, sys.psi1, FFTW_FORWARD, FFTW_MEASURE);
	sys.pb1 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi1, sys.psi1, FFTW_BACKWARD, FFTW_MEASURE);
	sys.pf2 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi2, sys.psi2, FFTW_FORWARD, FFTW_MEASURE);
	sys.pb2 = fftw_plan_dft_2d(sys.x.gridsize, sys.y.gridsize, sys.psi2, sys.psi2, FFTW_BACKWARD, FFTW_MEASURE);


	// Check for possible errors in memory allocation
	if((sys.psi1==NULL)||(sys.psi2==NULL)||(sys.consfield11==NULL)||(sys.k2==NULL)||(sys.x2==NULL)||(sys.extrapot==NULL)||(sys.x.x==NULL)||(sys.x.k==NULL)||(sys.y.x==NULL)||(sys.y.k==NULL))
		printf("Memory wrongly allocated!\n!\n!\n!\n!\n");

	double temp1,temp2;

	for(ii=0;ii<sys.gridsize;ii++){
		fscanf(file,"%lE\t%lE\n",&temp1,&temp2);
		sys.psi1[ii] = temp1 + I*temp2;
	}

	if(sys.currpoint>0){
		for(ii=0;ii<sys.gridsize;ii++){
			fscanf(file,"%lE\t%lE\n",&temp1,&temp2);
			sys.psi2[ii] = temp1 + I*temp2;
		}
	} else {
		for(ii=0;ii<sys.gridsize;ii++){
			sys.psi2[ii] = 0;
		}
	}

	// Close the file stream
	fclose(file);

	// Calculate the particle number
	//printf("npart: %E\n",densintegrate(sys));

	// Temp: Set Raman Detuning
	sys.ramandet = 2.*Pi*500E6;

	// Initialize auxilary variables
	// x-direction
	sys.x.losc = sqrt(hbar/(mass*sys.x.w));
	sys.x.tosc = 2.*Pi/sys.x.w;

	// y-direction
	sys.y.losc = sqrt(hbar/(mass*sys.y.w));
	sys.y.tosc = 2.*Pi/sys.y.w;

	// z-direction (intgr. out)
	sys.z.losc = sqrt(hbar/(mass*sys.z.w));

	// Chemical potential
	sys.mu = 1.47708846953*pow(sys.npart*as11/pow(sys.x.losc*sys.y.losc*sys.z.losc,1./3.),2./5.)*hbar*pow(sys.x.w*sys.y.w*sys.z.w,1./3.); // prefactor = 15^(2/5)/2

	// TF radii
	sys.x.rtf = sqrt(2.*sys.mu/(mass*sys.x.w*sys.x.w));
	sys.y.rtf = sqrt(2.*sys.mu/(mass*sys.y.w*sys.y.w));

	// Scaled parameters
	sys.mus = sys.mu*sys.tscale/hbar; // Calculate the scaled chemical potential
	sys.gs11 = 4*Pi*hbar*as11*sys.tscale/(mass * pow(sys.lscale,3.));
	sys.gs12 = 4*Pi*hbar*as12*sys.tscale/(mass * pow(sys.lscale,3.));
	sys.gs22 = 4*Pi*hbar*as22*sys.tscale/(mass * pow(sys.lscale,3.));

	// Some direction dependant variables
	// For x
	sys.x.rtfs = sys.x.rtf/sys.lscale;
	sys.x.dx = 2.*sys.x.s /(double)sys.x.gridsize;
	sys.x.dk = Pi/sys.x.s;

	// For y
	sys.y.rtfs = sys.y.rtf/sys.lscale;
	sys.y.dx = 2.*sys.y.s /(double)sys.y.gridsize;
	sys.y.dk = Pi/sys.y.s;

	// Initialize the grids
	for(ii=0;ii<sys.x.gridsize;ii++){
		sys.x.x[ii] = -sys.x.s + sys.x.dx*((double)ii+0.5);
		if(ii<sys.x.gridsize/2)
			sys.x.k[ii] = sys.x.dk*(double)ii;
		else
			sys.x.k[ii] = sys.x.dk*(double)(sys.x.gridsize-ii);
	}

	for(ii=0;ii<sys.y.gridsize;ii++){
		sys.y.x[ii] = -sys.y.s + sys.y.dx*(double)ii;
		if(ii<sys.y.gridsize/2)
			sys.y.k[ii] = sys.y.dk*((double)ii+0.5);
		else
			sys.y.k[ii] = sys.y.dk*(double)(sys.y.gridsize-ii);
	}

	// Initialize the squared grids
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			sys.k2[ii*sys.y.gridsize+jj] = sys.x.k[ii]*sys.x.k[ii]+sys.y.k[jj]*sys.y.k[jj];
			sys.x2[ii*sys.y.gridsize+jj] = sys.x.x[ii]*sys.x.x[ii]+(sys.y.w/sys.x.w)*(sys.y.w/sys.x.w)*sys.y.x[jj]*sys.y.x[jj];
		}
	}



	// Initialize the simulation constants
	sys.consx = -0.25*(1.*I+sys.dampfactor)*(mass * sys.x.w * sys.lscale * sys.lscale)/hbar * sys.x.w * sys.tscale * sys.dt; // Divided by 2 because we do 2 seperate space steps
	sys.consk = -0.5*(1.*I+sys.dampfactor)*hbar/(mass * sys.x.w * sys.lscale * sys.lscale) * sys.x.w * sys.tscale * sys.dt;
	sys.consc = -0.125*sys.consk;
	sys.consmu = (1.*I+sys.dampfactor)*sys.mus*sys.dt;
	//sys.consint11 = -0.5*0.75*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*sys.gs11/(pow(Pi,2./3.)*pow(sys.z.losc,4./3.)*pow(as11,1./3.))*sys.dt; // Divided by 2 because we do 2 seperate space steps

	// Interaction constants
	// double gavg = (sys.gs11+2.*sys.gs12+sys.gs22)/4.;
	// double asavg = (as11+2.*as12+as22)/4.;
	
	sys.consint11 = -0.5*(1.*I+sys.dampfactor)*sys.gs11*sys.dt;
	sys.consint12 = 0;//-0.5*(1.*I+sys.dampfactor)*0.25*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*(2*sys.gs12+gavg)/(pow(Pi,2./3.)*pow(sys.z.losc,4./3.)*pow(asavg,1./3.))*sys.dt;
	sys.consint22 = 0;//-0.5*(1.*I+sys.dampfactor)*0.25*pow(sys.lscale,2./3.)*pow(sys.npart,2./3.)*(2*sys.gs22+gavg)/(pow(Pi,2./3.)*pow(sys.z.losc,4./3.)*pow(asavg,1./3.))*sys.dt;

	// Coupling arrays
	//double raman1 = 1.5E-29 * 10./hbar; //d=1.5E-29 Cm * 10 V/m /hbar
	//double raman2 = 1.5E-29 * 10./hbar;

	//double complex coup11 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w * raman1*raman1/sys.ramandet; // dt* coefficient scaling * 
	//double complex coup12 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w * raman2*raman1/sys.ramandet;
	//double complex coup22 = (1.*I+sys.dampfactor)*sys.dt*sys.tscale*sys.x.w * raman2*raman2/sys.ramandet;

	//printf("Init constants\n");

	//double complex phase = 1.;

	// Arrays, since we want to add profile to these.
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			
			//phase = (sys.x.x[ii]+sys.y.x[jj]*I)/(20E-6/sys.lscale);
				
			//if(cabs(phase) > 1)
			//	phase = phase/cabs(phase);

			sys.consfield11[ii*sys.y.gridsize+jj] = 0;//coup11;
			sys.consfield21[ii*sys.y.gridsize+jj] = 0;//coup12*phase;
			sys.consfield12[ii*sys.y.gridsize+jj] = 0;//coup12*conj(phase);
			sys.consfield22[ii*sys.y.gridsize+jj] = 0;//coup22*phase*conj(phase);
		}
	}

	// Adding time-dependence
	for(ii=0;ii<sys.endpoint;ii++){
		sys.field1t[ii]=0.;
		sys.field2t[ii]=0.;
	}

	// Adding time-dependence
	//for(ii=0;ii<sys.endpoint;ii++){
	//	sys.field1t[ii] = 1;
	//	sys.field2t[ii] = 1;
	//}
	
	printf("consx: %E,%E\n",creal(sys.consx),cimag(sys.consx));
	printf("consk: %E,%E\n",creal(sys.consk),cimag(sys.consk));
	printf("consmu: %E,%E\n",creal(sys.consmu),cimag(sys.consmu));
	printf("consint12: %E,%E\n",creal(sys.consint12),cimag(sys.consint12));
	//printf("coup11: %E,%E\n",creal(coup11),cimag(coup11));
	//printf("RamanFreq: %E\n",raman1*raman2/sys.ramandet);

	//printf("List of constants:\nmu: %E\nnpart: %E\nx: %E\nk: %E\nint: %E\ncohl: %E\ndx: %E\ndy: %E\n",sys.mu,sys.npart,sys.consx,sys.consk,sys.consint11,(hbar/sqrt(2.*mass*sys.mu))/sys.lscale,sys.x.dx,sys.y.dx);

	return sys;
}


// Performs a single time step on the system
struct system timestep(struct system sys){
	int ii;
	
	double timecoeff = sys.field1t[sys.currpoint]*sys.field2t[sys.currpoint];

	//printf("Step: %da\nTimecoeff: %E\nDens: %E\nDens1: %E\tDens2: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys),densintegrate1(sys),densintegrate2(sys));

	#pragma omp parallel
	{

		struct matr m;
		double complex psi1,psi2;
		double temp;

		#pragma omp for
		for(ii=0;ii<sys.gridsize;ii++){
			psi1 = sys.psi1[ii];
			//psi2 = sys.psi2[ii];
		
			//m.matrix[0] = sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/(fabs(sys.y.x[ii%sys.y.gridsize]))+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize])+1.*sys.extrapot[ii]*sys.field1t[sys.currpoint];//+sys.consint12*psi2*conj(psi2))*temp+sys.consfield11[ii]*timecoeff;
			//m.matrix[1] = 0;//sys.consfield12[ii]*timecoeff;
			//m.matrix[2] = 0;//sys.consfield21[ii]*timecoeff;
			//m.matrix[3] = 0;//sys.consx * sys.x2[ii] + (sys.consint22*psi2*conj(psi2)+sys.consint12*psi1*conj(psi1))*temp+sys.consfield22[ii]*timecoeff;// Mf=0 doesn't coouple to magnetic field

			//expmatrwr(&m);

			//sys.psi1[ii] = m.matrix[0]*psi1+m.matrix[1]*psi2;
			//sys.psi2[ii] = m.matrix[2]*psi1+m.matrix[3]*psi2;
			sys.psi1[ii] = sys.psi1[ii]*cexp((sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/(fabs(sys.y.x[ii%sys.y.gridsize]))+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize])+sys.consx*2*sys.moddepth*sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize]*sys.field1t[sys.currpoint]) + sys.consmu);
		}
	}

	//printf("Step: %db\nTimecoeff: %E\nDens: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys));

	//printf("1. Step %d/%d;\t psi.dot.groundstate =  %E\tdens:%E\n",sys.currpoint,sys.endpoint,innerproductwithgroundstate(sys),densintegrate(sys));

	// Fourier transform of wave function
	fftw_execute(sys.pf1);
	//fftw_execute(sys.pf2);
	//printf("Step: %dc\nTimecoeff: %E\nDens: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys));
	//printf("DensIn: %E\n",densintegrate(sys));

	#pragma omp parallel for
	for(ii=0;ii<sys.gridsize;ii++){
		sys.psi1[ii] = sys.psi1[ii]*cexp(sys.consk * sys.k2[ii])/(double)sys.gridsize;
		//sys.psi2[ii] = sys.psi2[ii]*cexp(sys.consk * sys.k2[ii] - sys.consmu)/(double)sys.gridsize;
	}
	//printf("Step: %dd\nTimecoeff: %E\nDens: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys));
	//printf("DensUit: %E\n",densintegrate(sys));

	// Fourier transform back
	fftw_execute(sys.pb1);
	//fftw_execute(sys.pb2);

	//printf("DensUit: %E\n",densintegrate(sys));

	if(sys.dampfactor !=0)
		sys = normalize(sys);
	
	//printf("Step: %db\nTimecoeff: %E\nDens: %E\nDens1: %E\tDens2: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys),densintegrate1(sys),densintegrate2(sys));
	//printf("4. Step %d/%d;\t psi.dot.groundstate =  %E\n",sys.currpoint,sys.endpoint,innerproductwithgroundstate(sys));
	//printf("Step: %dd\nTimecoeff: %E\nDens: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys));
	#pragma omp parallel
	{

		struct matr m;
		double complex psi1,psi2;
		double temp;

		#pragma omp for
		for(ii=0;ii<sys.gridsize;ii++){
			psi1 = sys.psi1[ii];
			//psi2 = sys.psi2[ii];
		
			//m.matrix[0] = sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/(fabs(sys.y.x[ii%sys.y.gridsize]))+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize])+1.*sys.extrapot[ii]*sys.field1t[sys.currpoint];//+sys.consint12*psi2*conj(psi2))*temp+sys.consfield11[ii]*timecoeff;
			//m.matrix[1] = 0;//sys.consfield12[ii]*timecoeff;
			//m.matrix[2] = 0;//sys.consfield21[ii]*timecoeff;
			//m.matrix[3] = 0;//sys.consx * sys.x2[ii] + (sys.consint22*psi2*conj(psi2)+sys.consint12*psi1*conj(psi1))*temp+sys.consfield22[ii]*timecoeff;// Mf=0 doesn't coouple to magnetic field

			//expmatrwr(&m);

			//sys.psi1[ii] = m.matrix[0]*psi1+m.matrix[1]*psi2;
			//sys.psi2[ii] = m.matrix[2]*psi1+m.matrix[3]*psi2;
			sys.psi1[ii] = sys.psi1[ii]*cexp((sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/(fabs(sys.y.x[ii%sys.y.gridsize]))+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize])+sys.consx*2*sys.moddepth*sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize]*sys.field1t[sys.currpoint]) + sys.consmu);

		}
	}
	//printf("Step: %de\nTimecoeff: %E\nDens: %E\n\n",sys.currpoint,timecoeff,densintegrate(sys));
	//printf("5. Step %d/%d;\t %E\n",sys.currpoint,sys.endpoint,sys.extrapot[ii]*sys.field1t[sys.currpoint]);
	
	/*if(sys.dampfactor !=0)
		sys = normalize(sys);*/

	sys.currpoint++;

	// Return updated system
	return sys;
}

// Performs a single time step on the system, only for the FIRST component!
struct system imaginarytimestep(struct system sys){
	int ii;
	
	#pragma omp parallel for
	for(ii=0;ii<sys.gridsize;ii++){
		sys.psi1[ii] = sys.psi1[ii]*cexp(1./(1.*I+sys.dampfactor)*(sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/(fabs(sys.y.x[ii%sys.y.gridsize]))+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize]) + sys.consmu));
	}
	//printf("Densintegrate1: %E\n",densintegrate1(sys));
	// Fourier transform of wave function
	fftw_execute(sys.pf1);

	#pragma omp parallel for
	for(ii=0;ii<sys.gridsize;ii++){
			sys.psi1[ii] = sys.psi1[ii]*cexp(1./(1.*I+sys.dampfactor)*(sys.consk * sys.k2[ii]))/(double)sys.gridsize;
	//		sys.psi1[ii] = sys.psi1[ii]/sqrt((double)sys.gridsize);
	}
	//printf("Densintegrate2: %E\n",densintegrate1(sys));
	// Fourier transform back
	fftw_execute(sys.pb1);

	// Without this normalization the system will start in another state than the ground state and already contains both oscillations.
	//sys = normalize(sys);
	
	#pragma omp parallel for
	for(ii=0;ii<sys.gridsize;ii++){
		sys.psi1[ii] = sys.psi1[ii]*cexp(1./(1.*I+sys.dampfactor)*(sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/(fabs(sys.y.x[ii%sys.y.gridsize]))+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize]) + sys.consmu));
	}
	//printf("Densintegrate3: %E\n",densintegrate1(sys));

	sys.currpoint++;

	// Return updated system
	return sys;//normalize(sys);
}

// Normalize the wavefunction
struct system normalize(struct system sys){
	int ii; // Initialize iterators

	double invtotal = 1/sqrt(densintegrate(sys));

	#pragma omp parallel for
	for(ii=0;ii<sys.gridsize;ii++){
		sys.psi1[ii] *= invtotal;
		sys.psi2[ii] *= invtotal;	
	}

	return sys;
}

// Calculate the density integral
double densintegrate(struct system sys){
	int ii; // Initialize iterators

	double total = 0;
	for(ii=0;ii<sys.gridsize;ii++){
		total += creal(sys.psi1[ii]*conj(sys.psi1[ii])+sys.psi2[ii]*conj(sys.psi2[ii]));
	}
	return total*sys.x.dx*sys.y.dx;
}

// Calculate the density integral
double densintegrate1(struct system sys){
	int ii; // Initialize iterators

	double total = 0;
	for(ii=0;ii<sys.gridsize;ii++){
		total += creal(sys.psi1[ii]*conj(sys.psi1[ii]));
	}
	return total*sys.x.dx*sys.y.dx;
}

// Calculate the density integral
double densintegrate2(struct system sys){
	int ii; // Initialize iterators

	double total = 0;
	for(ii=0;ii<sys.gridsize;ii++){
		total += creal(sys.psi2[ii]*conj(sys.psi2[ii]));
	}
	return total*sys.x.dx*sys.y.dx;
}

double energy1(struct system sys){
	int ii; // Initialize iterators

	double total = 0;

	fftw_execute(sys.pf1);
	for(ii=0;ii<sys.gridsize;ii++){
		total += creal(-1.*I*((sys.psi1[ii]*conj(sys.psi1[ii])*(sys.consk * sys.k2[ii]))/(sys.dt*sys.gridsize)));
	}
	fftw_execute(sys.pb1);
	for(ii=0;ii<sys.gridsize;ii++){
		sys.psi1[ii] /= sys.gridsize;
		total += creal(-2.*I*((sys.psi1[ii]*conj(sys.psi1[ii])*(sys.consx * sys.x2[ii] + sys.consint11*sys.psi1[ii]*conj(sys.psi1[ii])/fabs(sys.y.x[ii%sys.y.gridsize])+sys.consc/(sys.y.x[ii%sys.y.gridsize]*sys.y.x[ii%sys.y.gridsize])+sys.consmu))/(sys.dt)));
	}
	//printf("energy: %E\n",total*sys.x.dx*sys.y.dx);
	return total*sys.x.dx*sys.y.dx;
}

double innerproductwithgroundstate(struct system sys){
	int ii,jj; // Initialize iterators

	double temp;
	double complex total = 0;

	double groundstate;

	double tempconst = sys.lscale*sqrt(pow(sys.mu,3./2.)/(3.*sqrt(3.*Pi)*pow(hbar*2.*Pi*104.,3./2.)*as11*sqrt(hbar/(mass*2.*Pi*104.))*sys.npart));
	for(ii=0;ii<sys.x.gridsize;ii++){
		for(jj=0;jj<sys.y.gridsize;jj++){
			groundstate = tempconst*pow(MAX(0.,1-(sys.x.x[ii]/sys.x.rtfs)*(sys.x.x[ii]/sys.x.rtfs)-(sys.y.x[jj]/sys.y.rtfs)*(sys.y.x[jj]/sys.y.rtfs)),3./4.);
			total += groundstate * (sys.psi1[ii*sys.y.gridsize+jj]+sys.psi2[ii*sys.y.gridsize+jj]);
		}
	}

	return cabs(total)*cabs(total)*sys.x.dx*sys.x.dx*sys.y.dx*sys.y.dx;
}


//
//	Operations on matrices below here.
//

// Call first. Calculates eigenvalues
void eigenval(struct matr *m)
{
	double complex trace = m->matrix[0]+m->matrix[3];
	double complex det = m->matrix[0]*m->matrix[3] - m->matrix[1]*m->matrix[2];
	m->eigenvalue[0] = (trace - csqrt(trace*trace - 4.*det))/2;
	m->eigenvalue[1] = (trace + csqrt(trace*trace - 4.*det))/2;

	if(m->eigenvalue[0]*m->eigenvalue[1] == NAN)
		printf("Eigenval NAN!\n");

	return;
}

// Calculates eigenvectors. Call after eigenval.
void eigenvec(struct matr *m)
{
	double complex temp1;

	temp1 = (m->eigenvalue[0] - m->matrix[0])/m->matrix[1];
	m->eigenvector[0] = 1/csqrt(1+temp1*conj(temp1));
	m->eigenvector[2] = temp1*m->eigenvector[0];

	temp1 = (m->eigenvalue[1] - m->matrix[0])/m->matrix[1];
	m->eigenvector[1] = 1/csqrt(1+temp1*conj(temp1));
	m->eigenvector[3] = temp1*m->eigenvector[1];
	
	if(m->eigenvector[0]*m->eigenvector[2] == NAN)
		printf("Eigenvec1 NAN!\n");

	if(m->eigenvector[1]*m->eigenvector[3] == NAN)
		printf("Eigenvec2 NAN!\n");

	return;
}

// Calculates the Matrix Exponential4. Step 1/100000;	 psi.dot.groundstate =  INF

void expmatr(struct matr *m)
{
	double complex exp1 = cexp(m->eigenvalue[0]);
	double complex exp2 = cexp(m->eigenvalue[1]);

	m->matrix[0] = m->eigenvector[0]*conj(m->eigenvector[0])*exp1+m->eigenvector[1]*conj(m->eigenvector[1])*exp2;
	m->matrix[1] = m->eigenvector[0]*conj(m->eigenvector[2])*exp1+m->eigenvector[1]*conj(m->eigenvector[3])*exp2;
	m->matrix[2] = m->eigenvector[2]*conj(m->eigenvector[0])*exp1+m->eigenvector[3]*conj(m->eigenvector[1])*exp2;
	m->matrix[3] = m->eigenvector[2]*conj(m->eigenvector[2])*exp1+m->eigenvector[3]*conj(m->eigenvector[3])*exp2;
	
	return;
}

// Wrapper to call neatly in code
void expmatrwr(struct matr *m){

	if(cabs(m->matrix[1]) == 0)
	{
		m->matrix[0] = cexp(m->matrix[0]);
		m->matrix[3] = cexp(m->matrix[3]);

	} else {
		eigenval(m);
		eigenvec(m);
		expmatr(m);
	}
	return;
}
