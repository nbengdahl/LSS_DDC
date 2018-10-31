clear all; close all; clc;
% ========================================================================
% Example of particle-based mass transfer schemes for simulating diffusion
%
% Note: this code requires the statistics and machine learning toolbox for
% the LSS (xf_mode=1) options, but the other modes do not
%
% Supplement to:
% Engdahl, Schmidt, and Benson (2019) "Accelerating and parallelizing 
%         Lagrangian simulations of mixing-limited reactive transport"
%         submitted to Water Resources Research
% ------------------------------------------------------------------------

% ========================================================================
%             >>>>>     Set user options      <<<<<    

np=2500;                % Number of particles
DDC=6;                  % Number of sub-domains (DDC only, set xf_mode=2)

pad_size=3;             % Size of domain pad (in # of StdDev's for xf_mode=[1 2], suggested value is 3)

xf_mode=2;      
% Options for x-fer mode (xf_mode): 
% 0 - Use the full matrix (slowest, same as DDC=1) 
% 1 - KDtree sparse looping (LSS)
% 2 - Domain decomposition (DDC)


%   --------------------------------------------------------------------
%             Suggested parameters for comparative runs
%   --------------------------------------------------------------------
% Set np=2500, then run
% 1. xf_mode=0
% 2. xf_mode=1
% 3. xf_mode=2, DDC=2
% 4. xf_mode=2, DDC=6
%
% -- Run 4 should be slightly quicker than run 2 for this setup, but both 
%     should be much faster than run 1 (6-8 times faster)
% -- Run 3 should be a slightly more than twice as fast as run 1
% ========================================================================


% *****           The remiander of the script is automated            *****
% *************************************************************************

rng(8936);               % Set random seed

% Initial condition
x_lims=[-25 25];        % Limits of domain
D0=1e0;                 % Diffusion coefficient
dt=0.1; nt=100;         % Time step and max simulation time 
Xp=zeros(np,3);
Xp(:,1)=x_lims(1) + rand(np,1).*(x_lims(2)-x_lims(1));
% Xp(:,1)=linspace(x_lims(1),x_lims(2),np);         % <= Uncomment for even spacing instead of random
Xp(:,2)=1.*(Xp(:,1)>=mean(x_lims));         % Conservative component 1
Xp(:,3)=1.*(Xp(:,1)<mean(x_lims));          % Conservative component 2


% Split diffusion into a deterministic part and a random part
D_RW=D0./2;     % Random walk part of diffusion/mixing
D_XF=D0./2;     % Mass transfer part of diffusion/mixing

dzRat=0;
switch xf_mode  % Sets up parameters for each x-fer mode
    case(0)
        % No setup needed
        xf_str='Brute force';
        ds=(x_lims(2)-x_lims(1))./np;               % Representative particle volume
    case(1)
        ds=(x_lims(2)-x_lims(1))./np;               % Representative particle volume
        dist=pad_size*sqrt(4*D_XF*dt);             % Max size for KDtree
        factor=ds/sqrt(8*pi*(1.0*D_XF)*dt);         
        denom=-8*(1.0*D_XF)*dt;
        xf_str='LSS';
    case(2)
        ds=(x_lims(2)-x_lims(1))./np;                       % Representative particle volume
        dzt=pad_size*sqrt(4*D_XF*dt);                       % Max size for KDtree
        d_cuts=linspace(x_lims(1),x_lims(2),DDC+1);       % End points of each subdomain
        dzRat=dzt/((x_lims(2)-x_lims(1))/DDC);            % Ratio of pad to subdomain size
        xf_str='DDC';
    otherwise
        error('Invalid x-fer mode');
end

if dzRat>0 
    fprintf('Ratio of ghost pad size to subdomain size: %3.3f \n',dzRat); 
    disp('                     (Suggested maximum is 0.2)');
end

Xp0=Xp; % Stor initial condition
nus=@(s,dt,D1,D2,n) exp(-(s.^2)/(4.*dt.*(D1 + D2)))./((4.*pi.*dt.*(D1 + D2)).^(n./2));
tic
for nstep=1:nt
    % Random diffusion
    Xp(:,1)=Xp(:,1) + sqrt(2.*D_RW.*dt).*randn(length(Xp(:,1)),1);
    R_out=find(Xp(:,1)>x_lims(2));  % Simple bounceback boundaries
    L_out=find(Xp(:,1)<x_lims(1));
    Xp(R_out,1)=x_lims(2) - (Xp(R_out,1)-x_lims(2));
    Xp(L_out,1)=x_lims(1) + (x_lims(1)-Xp(L_out,1));
    
    % X-fer selection block
switch xf_mode
    case(0) % Brute force, full matrix approach
        % Note: Case zero is s the same as case 3 with n_dom=1
        X=repmat(Xp(:,1)',np,1);
        S=abs(X-transpose(X));
        Nu=nus(S,dt,D_XF,D_XF,1);    
        Nu=Nu.*ds; 
        SNu=sum(Nu); % Sums for the normalization
        SNu(SNu<1e-16)=0; SNu(SNu==0)=1;    % Truncate small values and prevent dividing by zero
        Nu = Nu * diag(1./SNu);     % Normalize CLP        
        if any(any(isfinite(Nu)~=1))
            error('Error in colocation density normalization');
        end
        % Standard x-fer step for each component
        for km=1:2
        M=repmat(Xp(:,1+km)',np,1); 
        Md=M-transpose(M);       
        dM=((0.5).*(Md.*Nu));
        ddM=sum(dM,2);
        Xp(:,1+km)=Xp(:,1+km)+ddM(:);
        end
    case(1) % LSS - KDtree based looping via rangesearch  
       [idx, r]=rangesearch(Xp(:,1),Xp(:,1),dist,'BucketSize',10);
       for ni=1:np
       blah=idx{ni};     % This is the indices list of nearby particles to i
       s=r{ni};          % Associated radii
       s=s(blah>ni);
       apple=blah(blah>ni);
       if(isempty(apple)~=1)
          Ptot=factor*exp((s.^2)/denom);
          rescale=max(1,sum(Ptot)+factor);  
          order=randperm(length(apple));
          for j=1:length(apple)
            jpart=apple(order(j));    % Index of the current "other" particle
            vs_ds=Ptot(order(j))/rescale;
            for km=1:2
            dm = 0.5*(Xp(ni,1+km)-Xp(jpart,1+km))*vs_ds;
            %update particle masses both A and B 
            Xp(ni,1+km)=max(0,(Xp(ni,1+km)-dm));
            Xp(jpart,1+km)=max(0,(Xp(jpart,1+km)+dm));
            end
          end   
        end  % If statement for more than on b particle 
       end
             
    case(2) % DDC - Domain decomposition using DDC-n sub-domains
        Xp_in=Xp;       % INITIAL state

    for m=1:DDC       % Simple domain decomposition
        % Grab the particles within the subdomain, add pads to that if nonzero
        ID=find((Xp(:,1)>=(d_cuts(m)-dzt))&(Xp(:,1)<(d_cuts(m+1)+dzt)));

        xp=Xp_in(ID,:);  % Strict forward Euler
        nnp=length(ID);
        
        X=repmat(xp(:,1)',nnp,1);
        S=abs(X-transpose(X));
        Nu=nus(S,dt,D_RW,D_RW,1);    
        Nu=Nu.*ds; 
        SNu=sum(Nu);                        % Sums for the normalization
        SNu(SNu<1e-16)=0; SNu(SNu==0)=1;    % Truncate small values and prevent dividing by zero
        Nu = Nu * diag(1./SNu);             % Normalize probabilities
        if any(any(isfinite(Nu)~=1))
            error('Error in colocation density normalization');
        end
        % Some are ghost particles, so just update the non-ghosts 
        NGvec=1.*(Xp(ID,1)>=d_cuts(m))&(Xp(ID,1)<d_cuts(m+1));
        % Apply mass transfer step for each component
        for km=1:2
        M=repmat(xp(:,1+km)',nnp,1); 
        Md=M-transpose(M);       
        dM=((0.5).*(Md.*Nu));
        ddM=sum(dM,2);
        Xp(ID,1+km)=Xp(ID,1+km)+ddM(:).*NGvec; % Non-ghosts only
        end
    end
        
    otherwise
        error('Invalid x-fer mode');
        
end

% Reactions could then go here
    
end
runtime=toc;

Gau=@(x,D,t) 0.5.*erfc(-(x./sqrt(4.*D.*t)));
figure;
set(gcf,'position',[0.05 1 1 1].*get(gcf,'position'));
xc=linspace(x_lims(1),x_lims(2),100);
plot(xc,Gau(xc,D0,nt.*dt),'r'); hold on; grid on;
plot(Xp(:,1),Xp(:,2),'.');
plot(Xp(:,1),Xp(:,3),'.');
legend('Analytic','C1','C2');
xlabel('Distance [L]'); ylabel('Normalized concentration');
title(sprintf('Runtime %7.3e using %s scheme',runtime,xf_str));
