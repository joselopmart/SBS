%CON ESTE PROG EL CRITERIO QUE TOMAMOS ES PsL=1e-6*Ppo, (LIBRO AGRAWAL)
%EVOLUCION DE POTENCIA CON RESPECTO A Ppo
%Entre mas pequeño es delta es mas preciso los resultados solo que tarda
%mas, monitorear que PsL_Ppo sea 1E-6 para cumplir con el Criterio
tic
clc;

w=0:12.5:25;%5;%0:10:20;%rango distancia en km
%Ppo=0:0.1:PpoMAX;
%Ppo=0.001:0.001:0.021;%Pot. en watts
Ppo=0.0005:0.001:0.0045;  %0.0001:0.004 ;%Pot. en watts .0051
 
PpF=[];
PsF=[];
for i=1:1:length(Ppo)
  
  PsL_Ppo=0.8; %0.8, 
  DELTA=0.000001;%OJO EL VALOR DE DELTA DEBE SER LO MAS PEQUEÑO POSIBLE, CON UN DELTA=0.000001 LA GRAFICA TIENE BUENA NITIDEZ PERO SE TARDA MUCHO TIEMPO
  %de preferencia Un decimo de la minima potencia de bombeo
  K= 0.53;%0.8;%iniciamos con Pso=0.53Ppo
  while (PsL_Ppo>0.0000011) 
     
     K=K-DELTA;  %vamos disminuyendo hasta el CRITERIO que la PsL=1e-6*Ppo  
     x0=[Ppo(i) K*Ppo(i)];%condiciones iniciales
  
    [w,x]=ode45('acopB',w,x0);
 
    PsL=x(length(x(:,2)),2);
    PsL_Ppo=PsL/Ppo(i);

  end
  PsL_Ppo
  K
 
  PpF(i)=x(length(x(:,1)),1);
  PsF(i)=x(1,2);
end
PsL_Ppo

%hold on;
 %grid on;
%plot(w,x(:,1),'k');
%plot(w,x(:,2),'r');

%figure;

 

hold on; 
 grid on;
%plot(Ppo/1e-3,PpF/1e-3,'k');
%plot(Ppo/1e-3,PsF/1e-3,'r');
Ppo_dBm=10*log10(Ppo/1e-3);
PpF_dBm=10*log10(PpF/1e-3);
PsF_dBm=10*log10(PsF/1e-3);
plot(Ppo_dBm,PpF_dBm,'k');
plot(Ppo_dBm,PsF_dBm,'r'); 

 title('Evol de las potencias respecto a Pin');
 xlabel('Pin(dBm)');
 ylabel('PpL,Pso(dBm)');
 %gtext('Forward (Negro), Backward (Rojo)');  
      

 toc

