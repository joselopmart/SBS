#include<iostream>
#include<cuda.h>
#include<math.h>
#include <time.h>


/* 1-  nvcc  acopladas_B3-2.cu -o acopladas_B3-2
  
   2-./acopladas_B3-2

We are using Dormand-Prince Method based on  http://depa.fquim.unam.mx/amyd/archivero/DormandPrince_19856.pdf   
*/


using namespace std;

__global__  void suma(int *a,int *b,int *c)
{
  int id=threadIdx.x;
  c[id]=a[id]+b[id];
};



__global__  void  Resuelve(double *dev_w, double *dev_Ppo, double *dev_PsL_Ppo,double *dev_K,double *dev_Delta,double *dev_PpF,double *dev_PsF)
{
 int id=threadIdx.x;
 double x0[2];double y1[3],y2[3];
 double eps=0.000001;
 double t0=0;
    double h=0.01;
    double hmin=0.00001;
    double hmax=0.1;
    
    double gB=4e-14;
    double Aeff=85e-18;
    double a=0.2/4.343;
    double b=gB/(1.2*Aeff);
    double K=0.53;
    
    double PsL_Ppo=0.8;
    
    int c=0;
    
       
  while (PsL_Ppo>0.0000011)
 {
        dev_K[id]=dev_K[id]-dev_Delta[id];       
       // K=K-0.000001;
       K=dev_K[id];
       x0[0]=dev_Ppo[id];                   //x0[0]=0.0005;
       x0[1]=K*x0[0];                     //x0[1]=dev_K[id]*dev_Ppo[id];        
       
     
     
     y1[0]=x0[0]; y1[1]=x0[0];
     y2[0]=x0[1];  y2[1]=x0[1];
     c=c+1;
     t0=0;
     h=0.01;
     
     while(t0<=25)
    {
        ///////////////
        
        
         double k11=-(a)*y1[0]-(b)*y1[0]*y2[0];
         double k12=(a)*y2[0]-(b)*y1[0]*y2[0];
         k11=k11*h;
         k12=k12*h;
         
         
         double ty1_k11=y1[0]+(k11)*(1.0/5);
         double ty2_k12=y2[0]+(k12)*(1.0/5);
         
         double k21=-(a)*ty1_k11-(b)*ty1_k11*ty2_k12;
         double k22=(a)*ty2_k12-(b)*ty1_k11*ty2_k12;
         
         k21=k21*h;
         k22=k22*h;
         
         
         double ty1_k21=y1[0]+(3.0/40)*(k11)+(9.0/40)*(k21);
         double ty2_k22=y2[0]+(3.0/40)*(k12)+(9.0/40)*(k22);
         
         double k31=-(a)*ty1_k21-(b)*ty1_k21*ty2_k22;
         double k32=(a)*ty2_k22-(b)*ty1_k21*ty2_k22;
         
         k31=k31*h;
         k32=k32*h;
         
         
         double ty1_k31=y1[0]+(44.0/45)*k11-(56.0/15)*k21+(32.0/9)*(k31);
         double ty2_k32=y2[0]+(44.0/45)*k12-(56.0/15)*k22+(32.0/9)*(k32);
         
         
         double k41=-(a)*ty1_k31-(b)*ty1_k31*ty2_k32;
         double k42=(a)*ty2_k32-(b)*ty1_k31*ty2_k32;
         
         
         k41=k41*h;
         k42=k42*h;
         
         
         double ty1_k41=y1[0]+(19372.0/6561)*k11-(25360.0/2187)*k21+(64448.0/6561)*(k31)-(212.0/729)*(k41);
         double ty2_k42=y2[0]+(19372.0/6561)*k12-(25360.0/2187)*k22+(64448.0/6561)*(k32)-(212.0/729)*(k42);
         
         double k51=-(a)*ty1_k41-(b)*ty1_k41*ty2_k42;
         double k52=(a)*ty2_k42-(b)*ty1_k41*ty2_k42;
         
         k51=k51*h;
         k52=k52*h;
         
         double ty1_k51=y1[0]+(9017.0/3168)*k11-(355.0/33)*k21-(46732.0/5247)*(k31)+(49.0/176)*(k41)-(5103.0/18656)*(k51);
         double ty2_k52=y2[0]+(9017.0/3168)*k12-(355.0/33)*k22-(46732.0/5247)*(k32)+(49.0/176)*(k42)-(5103.0/18656)*(k52);
         
         double k61=-(a)*ty1_k51-(b)*ty1_k51*ty2_k52;
         double k62=(a)*ty2_k52-(b)*ty1_k51*ty2_k52;
         
         k61=k61*h;
         k62=k62*h;
         
         
         double ty1_k61=y1[0]+(35.0/384)*k11+(500.0/1113)*(k31)+(125.0/192)*(k41)-(2187.0/6784)*(k51)+(11.0/84)*(k61);
         double ty2_k62=y2[0]+(35.0/384)*k12+(500.0/1113)*(k32)+(125.0/192)*(k42)-(2187.0/6784)*(k52)+(11.0/84)*(k62);
         
         double k71=-(a)*ty1_k61-(b)*ty1_k61*ty2_k62;
         double k72=(a)*ty2_k62-(b)*ty1_k61*ty2_k62;
         
         k71=k71*h;
         k72=k72*h;
         
         double tmpy1=y1[0]+((35.0/384)*k11+(500.0/1113)*k31+(125.0/192)*k41-(2187.0/6784)*k51+(11.0/84)*k61);
         double tmpy2=y2[0]+((35.0/384)*k12+(500.0/1113)*k32+(125.0/192)*k42-(2187.0/6784)*k52+(11.0/84)*k62);
         
         // double tmpz1=y1[0]+(5179.0/57600)*k11+(7571.0/16695)*k31+(393.0/640)*k41-(92097.0/339200)*k51+(187.0/2100)*k61+(1.0/40)*k71;
         double tmpz2=y2[0]+(5179.0/57600)*k12+(7571.0/16695)*k32+(393.0/640)*k42-(92097.0/339200)*k52+(187.0/2100)*k62+(1.0/40)*k72;
         
         double err=abs(tmpy2-tmpz2);
         
         double s=pow((eps*h)/(2.0*err),1.0/5);
         
         
         double h1=s*h;
         
         if (h1<hmin)
         h1=hmin;
         else if(h1>hmax) h1=hmax;
         
         
         t0=t0+h;
         y1[0]=tmpy1;
         y2[0]=tmpy2;
         h=h1;
        
        ///////////////
        
        
      
    }  //fin del for
    
     double PsL=y2[0];
      //dev_PsL_Ppo[id]=double(PsL)/dev_Ppo[id]; 
       PsL_Ppo=double(PsL)/dev_Ppo[id];
       
} //fin del while tolerancia
 
 
 dev_PpF[id]=y1[0]; 
 dev_PsF[id]=y2[1];
  
 //dev_PpF[id]=10*log10(y1[0]/1e-3);
 //dev_PsF[id]=10*log10(y2[1]/1e-3)  ; 
   
   
 //dev_PpF[id]=8; 
 //dev_PsF[id]=9;
   
 
};



int main()
{
  double DELTA;
  double *host_w,*host_Ppo,*host_PpF,*host_PsF,*host_PsL_Ppo,*host_K,*host_Delta;
  double *dev_w,*dev_Ppo,*dev_PpF,*dev_PsF,*dev_PsL_Ppo,*dev_K,*dev_Delta;
  
  
  DELTA=0.000001;
  
  double incrementow=12.5;
  double iniciow=0.0;
  double finalw=25.0;
  int Nw=ceil((finalw-iniciow)/incrementow); 
  
  
  host_w=new double[Nw];
  Nw=Nw+1;
  
   for (int i=0;i<Nw;i++)
   {
      
      host_w[i]=iniciow;
      iniciow=iniciow+incrementow;
  };
  
  
  
  
  double incrementoPpo=0.0001;
  double inicioPpo=0.0005;
  double finalPpo=0.0045;
  int N=ceil((finalPpo-inicioPpo)/incrementoPpo); 
  N=N+1;
  host_Ppo=new double[N];  
  
  host_PsL_Ppo=new double[N];  
  host_K= new double[N];
  host_Delta=new double[N];
  host_PpF=new double[N];
  host_PsF=new double[N];
  
  
  
  for (int i=0;i<N;i++)
   {
      
      host_PpF[i]=0.0;
      host_PsF[i]=0.0;
      host_PsL_Ppo[i]=0.8;
      host_K[i]=0.53;
      host_Delta[i]=DELTA;
      host_Ppo[i]=inicioPpo;
      inicioPpo=inicioPpo+incrementoPpo;
  };
  
  
  int memw=sizeof(double)*Nw;
  int mem=sizeof(double)*N;
  
  cudaMalloc((void **)&dev_Ppo,mem);
  cudaMalloc((void **)&dev_w,memw);
  cudaMalloc((void **)&dev_PsL_Ppo,mem);
  cudaMalloc((void **)&dev_K,mem);
  cudaMalloc((void **)&dev_Delta,mem);  
  cudaMalloc((void **)&dev_PpF,mem); 
  cudaMalloc((void **)&dev_PsF,mem);
  
  
  cudaMemcpy(dev_w,host_w,memw,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Ppo,host_Ppo,mem,cudaMemcpyHostToDevice); 
  cudaMemcpy(dev_PsL_Ppo,host_PsL_Ppo,mem,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_K,host_K,mem,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Delta,host_Delta,mem,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_PpF,host_PpF,mem,cudaMemcpyHostToDevice); 
  cudaMemcpy(dev_PsF,host_PsF,mem,cudaMemcpyHostToDevice); 
  
 
  cudaEvent_t start,stop;
   float time;
  
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start,0);
  
  
  Resuelve<<<1,N>>>(dev_w,dev_Ppo,dev_PsL_Ppo,dev_K,dev_Delta,dev_PpF,dev_PsF);
  
  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time,start,stop);
  cout<<"tiempo : "<<time/1000<<endl;
  printf("\nelapsedTime in ms= %f:\n",time);
  
  cudaMemcpy(host_K,dev_K,mem,cudaMemcpyDeviceToHost);
  cudaMemcpy(host_PsL_Ppo,dev_PsL_Ppo,mem,cudaMemcpyDeviceToHost);
  cudaMemcpy(host_PpF,dev_PpF,mem,cudaMemcpyDeviceToHost);
  cudaMemcpy(host_PsF,dev_PsF,mem,cudaMemcpyDeviceToHost);
  

  cout<<endl;
  cout<<N;
  cout<<endl;
  
  cout<<"PpF"<<endl;
  for(int i=0;i<N;i++)
  {
     cout<<" "<<host_PpF[i]<<" ";
  };
  
  cout<<endl;
  cout<<N;
  cout<<endl;
    cout<<"PsF"<<endl;
  
  for(int i=0;i<N;i++)
  {
     cout<<" "<<host_PsF[i]<<" ";
  };
  
  cout<<endl;
  
  cudaFree(dev_Ppo);
  cudaFree(dev_w);
  cudaFree(dev_K);
  cudaFree(dev_PsL_Ppo);
  cudaFree(dev_Delta);
  cudaFree(dev_PpF);
  cudaFree(dev_PsF);
  
  
  
 
 
  delete [] host_Ppo;
  delete [] host_w; 
  delete [] host_K; 
  delete [] host_PsL_Ppo; 
  delete [] host_Delta;
  delete [] host_PpF;
  delete [] host_PsF;
  
  
};

