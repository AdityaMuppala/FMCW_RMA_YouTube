%-- Code written by Aditya Varma Muppala for YouTube tutorial on FMCW Range Migration.
%-- A very similar code is also available on my GitHub to go with the paper titled: "Fast-Fourier Time-Domain SAR Reconstruction for
%-- Millimeter-Wave FMCW 3-D Imaging". Last edited on 03/05/2024.
%-- The data can be downloaded from google drive using the following link: https://drive.google.com/file/d/1AnGdp6I1WtOdJQiCnRePAfU8fW8nTHk3/view?usp=sharing
%-- Extract the zip file into a folder titled "Imaging_raw_data" and place in the same path as the MATLAB codes.

clc
clear
clear path

%-- Radar System Operational Parameters
fBW = 8e9;                 % bandwidth
fc = 79.6e9;                 % carrier frequency
wc = 2*pi*fc;
c  = 3e8;                  % RF propagation speed
theta_b = 60;              % Antenna Beamwidth in degrees

wBW = 2*pi*fBW;
RR = c/(2*fBW);

%-- Fast-Time domain parameters and arrays
fs = 50*1e6;                % Sampling rate;
T = 48*1e-6;                % in microseconds. Last 2 microseconds are dead zone

N = fs * T;                 % number of fast-time samples
gamma = -wBW/T;              % Chirp rate (System I used performs a Down-chirp! Hence the negative sign)

dt = 1/fs;
Tp  = (N-1)*dt;             % Chirp pulse duration
t = linspace(-Tp/2,Tp/2,N); % time array for data acquisition

range = 0:RR:RR*(N-1);

%-- Slow-Time domain parameters and arrays
spacing = 1.6e-3;
xp = -200e-3:spacing:200e-3;
yp = -200e-3:spacing:200e-3;
zp = 0;

Mx = length(xp);
My = length(yp);
M = Mx*My;                 % Number of slow time measurements

% Raw data acquisition (Try different data sets available in Imaging_new_data folder, but change the background file as well.)
% fileID = fopen('Imaging_raw_data/SiemensStencil_Measurement_1.ch0','rb');
% fileID = fopen('Imaging_raw_data/MStencil_Measurement_1.ch0','rb');
% fileID = fopen('Imaging_raw_data/Pistol_Measurement_1.ch0','rb');
% fileID = fopen('Imaging_raw_data/PistolStencilBox_Measurement_1.ch0','rb');
fileID = fopen('Imaging_raw_data/Mannequin_Measurement_1.ch0','rb');
ch1 = fread(fileID,'double');
fclose(fileID);

% fileID2 = fopen('Imaging_raw_data/Bkg_SiemensStencil_Measurement_1.ch0','rb');
% fileID2 = fopen('Imaging_raw_data/Bkg_MStencil_Measurement_1.ch0','rb');
% fileID2 = fopen('Imaging_raw_data/Bkg_Pistol_Measurement_1.ch0','rb');
% fileID2 = fopen('Imaging_raw_data/Bkg_PistolStencilBox_Measurement_1.ch0','rb');
fileID2 = fopen('Imaging_raw_data/Bkg_Mannequin_Measurement_1.ch0','rb');
ch1_bkg = fread(fileID2,'double');
fclose(fileID2);

% Re-shaping to fast time X slow time array
valid_samples_per_trigger = fs*T;
ch1b = squeeze(reshape(ch1,valid_samples_per_trigger,[],M));

sdata_time = ch1b-ch1_bkg; % Background subtraction

% sdata_time = sdata_time.*hanning(valid_samples_per_trigger); % Windowing in time (not required)

%% Calibration step 2

% Phase center calibration. See Section 2 of paper given above.
del_lt = 19.1e-2; % Obtained by sweeping
exp_shift = transpose(exp(-1i*(wc*2*del_lt/c-2*gamma*del_lt*t/c)));
sdata_time2 = sdata_time.*exp_shift;

%% Rearranging raster scanned data to XY - grid

disp('Reformatting raster scan to XY grid');
s_b = zeros(Mx,My,N);
main_ind = 1;

for Myind = My:-1:1
    if rem(Myind,2) == 0
        for Mxind = 1:Mx
            s_b(Mxind,Myind,:) = sdata_time2(:,main_ind);
            main_ind = main_ind+1;
        end
    else
        for Mxind = Mx:-1:1
            s_b(Mxind,Myind,:) = sdata_time2(:,main_ind);
            main_ind = main_ind+1;
        end
    end
end

%% Reconstruction - Precomputing K-space

Kt = gamma*t/c+wc/c;
k_limit = (2*wc/c)*sind(theta_b/2); % Light cone boundary

dkx = 2*pi/(2*max(xp));
dky = 2*pi/(2*max(yp));
kx = dkx*(-(Mx-1)/2:(Mx-1)/2);
ky = dky*(-(My-1)/2:(My-1)/2);

% Truncating data that lies inside antenna beam light cone
kx_trunc_ind = abs(kx)<=k_limit;
ky_trunc_ind = abs(kx)<=k_limit;
kx_trunc = kx(kx_trunc_ind);
ky_trunc = ky(ky_trunc_ind);

Mxn = length(kx_trunc); Myn = length(ky_trunc);

kx3D = repmat(kx_trunc.',[1,Myn,N]);
ky3D = repmat(ky_trunc,[Mxn,1,N]);
Kt1D(1,1,:) = Kt;
Kt3D = repmat(Kt1D,[Mxn,Myn,1]);
kz3D = sqrt(4*Kt3D.^2-kx3D.^2-ky3D.^2);

kz1D_uni = kz3D((Mxn+1)/2,(Myn+1)/2,:); % Uniform sampling of Kz taken along Kz axis
kz3D_uni = repmat(kz1D_uni,[Mxn,Myn,1]);

indX_range = 1:Mx; indY_range = 1:Mx;
truncX_range = indX_range(kx_trunc_ind);
truncY_range = indY_range(ky_trunc_ind);

%{
%% Visualizing K-space
%-- Choosing the ky = 0 cut

kx_temp = kx3D(:,107,:);
ky_temp = ky3D(:,107,:);
kz_temp = kz3D(:,107,:);
kzUni_temp = kz3D_uni(:,107,:);

figure(2); clf
scatter3(kx_temp(1:100:end),ky_temp(1:100:end),kz_temp(1:100:end)); hold on
scatter3(kx_temp(1:100:end),ky_temp(1:100:end),kzUni_temp(1:100:end))

xlabel('k_x');
ylabel('k_y');
zlabel('k_z');
view([180 0])
set(gcf,'position',[-850,50,800,800])
set(gca,'FontSize',20)
set(gca,'GridAlpha',1)
set(gca,'GridLineStyle','--')
set(gca,'fontname','ariel')
set(gcf,'color','w');

%}

%% Reconstruction - Algorithm

tic
disp('Starting algorithm. Est. time: 30 seconds');

%-- Step 1: First 2D FFT
disp('First 2-D FFT');
s_B = fty(ftx(s_b));

%--Step 2: Non-uniform to Uniform sampling (Stolt interpolation)
disp('Stolt Interpolation');
s_B2 = zeros(Mxn,Myn,N);

for indX = 1:Mxn
    for indY = 1:Myn
        s_B2(indX,indY,:) = interp1(squeeze(kz3D(indX,indY,:)),squeeze(s_B(truncX_range(indX),truncY_range(indY),:)),kz1D_uni);
    end
end

s_B2(isnan(s_B2))=0; % Truncating extrapolated values lying outside the grid to zero.

%--Step 3: 3-D IFFT
disp('Last 3-D IFFT');

f_hat = fft(ifty(iftx(s_B2)),[],3); % fft in Z is used instead of ifft due to time convention in t from measurement.

time = toc;
disp(['Total computation time: ',num2str(time), 'seconds']);

%% Plotting Final image

dx = 2*pi/(max(2*kx_trunc));
dy = 2*pi/(max(2*ky_trunc));
dz = 2*pi/(max(kz3D((Mxn+1)/2,(Myn+1)/2,:))-min(kz3D((Mxn+1)/2,(Myn+1)/2,:)));

xIm = dx*(-(Mxn-1)/2:(Mxn-1)/2);
yIm = dy*(-(Myn-1)/2:(Myn-1)/2);
distZ = dz*(1:N);

[X,Y] = ndgrid(xIm,yIm);
maxAll = max(abs(f_hat(:)));

for zInd = 1:length(distZ)
    zInd
    f = sum(abs(f_hat(:,:,zInd)),3);
    figure(1); clf
    surf(X*100,Y*100,abs(f/maxAll));
    clim([0 1]);
    shading interp;
    xlabel('X (cm)');
    ylabel('Y (cm)');
    title(['Z distance: ',num2str(distZ(zInd))]);
    hold on;
    axis equal; axis tight
    colormap gray;
    view([0 90])
    colorbar
    xlim([-20 20])
    ylim([-20 20])
    set(gcf,'position',[-850,50,800,800])
    set(gca,'FontSize',20)
    set(gca,'GridAlpha',1)
    set(gca,'GridLineStyle','--')
    set(gca,'fontname','ariel')
    set(gcf,'color','w');
    w = waitforbuttonpress;
end