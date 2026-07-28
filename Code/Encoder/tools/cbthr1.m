function masklev = cbthr1(cbpower,fc)

% cbthr1
% Critical band thresholds
% Ambi's method

cbands = length(cbpower);
%fp = fc/1000; % calculate centre frequencies in Bark
%bark = 13.0*atan(0.76*fp) + 3.5*atan((fp.^2)/(7.5*7.5));
bark = (1:cbands);

% Threshold in Quiet is calculated as follows
fz=fc/1000;
thresq=3.64*(fz.^(-0.8))-6.5*exp(-0.6*((fz-3.3).^2))+0.001*(fz.^4);

% Calculate the masking levels using an auditory masking model
for L=1:cbands %calculate the masking for critical band L (maskee)
   sum1=0;
   % calculate masking contribution from all other nodes to node L and sum
   for k=1:cbands, % masker
      vs=-2.025 + 0.825*cbpower(L); % maximum self masking level
      delta = bark(k)-bark(L); % Separation between the masker & maskee
      if ((delta >= -3)  & (delta <-1))
         vf= (17*delta -0.4*cbpower(L) +11);
      elseif ((delta >= -1) & (delta <0))
         vf = ((0.4*cbpower(L) +6)*delta);
      elseif ((delta >=0) & (delta <1))
         vf=-17*delta;
      elseif ((delta >=1) & (delta < 8))
         vf=-17*delta +0.15*(delta-1)*cbpower(L);
      else
         vf=0.000000001; %for values of delta below -3 and above 8 barks
         vs=0.000000001;
      end
      M=vs+vf;
      sum1=sum1+10^(M/20);
   end
   masklev(L)=20*log10(sum1)+ thresq(L);
end
