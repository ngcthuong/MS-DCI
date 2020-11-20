%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% @article{Canh2018_MSCSNet,
%   title={Multi-Scale Deep Compressive Sensing Network},
%   author={Thuong, Nguyen Canh and Byeungwoo, Jeon},
%   conference={IEEE International Conference on Visual Comunication and Image Processing},
%   year={2018}
% }
% by Thuong Nguyen Canh (9/2018)
% ngcthuong@gmail.com
% https://github.com/AtenaKid
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You need to install Matconvnet in order to run this code

warning('off','all')
addpath('D:\matconvnet-1.0-beta25\matlab\mex');
addpath('D:\matconvnet-1.0-beta25\matlab\simplenn');
% addpath('D:\matconvnet-1.0-beta25\matlab');

addpath('.\utilities');
folderTest  = 'Classic13_512';
networkTest = { 'SS-DCS1' 'SS-DCS2' 'SS-DCS3'};  % 102 

showResult  = 0;
writeRecon  = 1;
featureSize = 64;
blkSize     = 32;
isLearnMtx  = [1, 0];

network     = networkTest{3};
padSize     = [1, 1, 2]; 
subrate_all = [0.1:0.1:0.3];
rangeSize   = [0.15 0.3 0.25];

for subRate_id = 1:1:3
    subRate = subrate_all(subRate_id); 
    modelName   = [network '_r' num2str(subRate)]; %%% model name
    
    data = load(fullfile('models', network ,[modelName,'.mat']));
    net  = dagnn.DagNN.loadobj(data.net);
    if strcmp(network,'CSNet')
        net.renameVar('x0', 'input');
        net.renameVar('x12', 'prediction');
    else
        net.removeLayer(net.layers(end).name) ;
    end
    
    net.mode = 'test';
    net.move('gpu');
    
    %%% read images
    ext         =  {'*.jpg','*.png','*.bmp', '*.pgm', '*.tif'};
    filePaths   =  [];
    
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})) );
    end
    
    PSNRs_CSNet = zeros(1,length(filePaths));
    SSIMs_CSNet = zeros(1,length(filePaths));
    
    count = 1;
    allName = cell(1);
    
   
    for i = 1:1
        
        %%% read images
        image              = imread(fullfile('testsets', folderTest, filePaths(i).name));
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        allName{count}     = nameCur;
        
        if size(image,3) == 3
            image = modcrop(image,32);
            image = rgb2ycbcr(image);
            image = image(:,:,1);
        end
        label = im2single(image);
        if mod(size(label, 1), blkSize) ~= 0 || mod(size(label, 2), blkSize) ~= 0
            continue
        end
        
        input = label;
        input = gpuArray(input);
        
        tic
        net.eval({'input', input}) ;
        time(i) = toc;
        out1 = net.getVarIndex('prediction') ;
        output = gather(squeeze(gather(net.vars(out1).value)));
        
        
        %% Check the sampling matrix
        Phi = squeeze(gather(net.params(9).value));
        tLegend = {'1Low-Low', '2Low-High', '3High-Low', '4High-High'};
        
        [w, h, c, k] = size(Phi);
        k0 = floor(sqrt(k));
%         k0 = 5; 
        
        bigMtx = [];
        
         
        count = 1; 
        
        if subRate == 0.2
            color_range = [0.3, 0.3; 0.3, 0.06];
        end
        
        for idx1 = 1:1:2
            tmpBig = [];
            for idx2 = 1:1:2
                idx = (idx1-1) * 2 + idx2;
                m1 = 0; m2 = 0;
                pr = figure(idx);
                mtx = [];
                for i0 = 1:1:k0
                    im = [];
                    %count = 1; 
                    for i1 = 1:1:k0
                        i2 = (i0 - 1)*k0 + i1;                        
                        Phi_i = Phi(:, :, idx, i2);
                        
                        figure('Position', [10 10 900 600]);
                        f1 = imshow(Phi_i,[-color_range(idx1, idx2), color_range(idx1, idx2)]); axis square;axis off; 
                        %colormap(parula);
                        
                        m1(count) = min(Phi_i(:)); 
                        m2(count) = max(Phi_i(:)); 
                        count = count + 1; 
                        if i1 == 1
                            im = padarray(export_fig, [0, padSize(subRate_id)], 255, 'post');                             
                        else                            
                            im  = cat(2, im, padarray(export_fig, [0, padSize(subRate_id)], 255, 'post')); 
                        end
                        close all; 
                        
                    end                    
                    im = padarray(im, [padSize(subRate_id), 0], 255, 'post');      
                    mtx = cat(1, mtx, im);    
                    
                end             % End a subband 
                
                imshow(mtx, []); %title([tLegend{idx}]);
                im = export_fig; %color bar; 
                
                imwrite(im, ['mtxResults_grey\' tLegend{idx} '_meas_' modelName ...
                    '_rate' num2str(subRate) '_'  num2str(color_range(idx1, idx2)) '.png']);
                
                tmpBig = cat(2, tmpBig, padarray(im, [0, padSize(subRate_id) + 2], 255, 'post')); 
            end
            
            tmpBig = padarray(tmpBig, [padSize(subRate_id) + 2, 0], 255, 'post');             
            bigMtx = cat(1, bigMtx, tmpBig);
        end
        
        
        pr = figure(6); imshow(bigMtx);
        im = export_fig; 
        imwrite(im, ['mtxResults_grey\0meas_' modelName '_rate' num2str(subRate) '_'  num2str(rangeSize(subRate_id)) '.png']);
        saveas(pr, ['mtxResults_grey\0meas_' modelName '_rate'  num2str(subRate) '_'  num2str(rangeSize(subRate_id)) '.fig']);
    end
end
