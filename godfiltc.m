%Func?ia de filtrare a frecven?elor nedorite
%Filtru Band-Pass, valorile filtrului fiind ajustate în func?ie de necesitate
function [x]=doit(t,Drp2,Fs) 
Fs = Fs; 			    % Sampling Frequency
Fstop1 = 0.5;             % First Stopband Frequency
Fpass1 = 1;               % First Passband Frequency
Fpass2 = 40;              % Second Passband Frequency
Fstop2 = 42;              % Second Stopband Frequency
Dstop1 = 0.0001;          % First Stopband Attenuation
Dpass  = 0.057501127785;  % Passband Ripple
Dstop2 = 0.0001;          % Second Stopband Attenuation
dens   = 20;              % Density Factor
 
% Calculate the order from the parameters using FIRPMORD.
[Ns, Fo, Ao, W] = firpmord([Fstop1 Fpass1 Fpass2 Fstop2]/(Fs/2), [0 1...
                          0], [Dstop1 Dpass Dstop2]);
% Calculate the coefficients using the FIRPM function.
b3  = firpm(Ns, Fo, Ao, W, {dens});
Hd3 = dfilt.dffir(b3);
x=filter(Hd3,Drp2);

