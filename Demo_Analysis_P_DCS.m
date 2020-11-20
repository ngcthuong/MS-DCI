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
addpath('D:\matconvnet-1.0-beta25\matlab\mex');     % Link to your Matconvnet Mex file
% addpath('matconvnet-1.0-beta25\matlab');

addpath('.\utilities');
folderTest  = 'Classic13_512';
% networkTest = {'MS-CSNet1' 'MS-CSNet2' 'MS-CSNet3'};      % 10

networkTest = { 'P-DCS1' 'P-DCS2' 'P-DCS3'};

showResult  = 0;
writeRecon  = 1;
featureSize = 64;
blkSize     = 32;
isLearnMtx  = [1, 0];

for netId = [1]
network     = networkTest{netId};

for subRate = [0.1:0.1:0.3]
    modelName   = [network '_r' num2str(subRate)]; %%% model name
    
    data = load(fullfile('models', network ,[modelName,'.mat']));
    net  = dagnn.DagNN.loadobj(data.net);
    if strcmp(network,'CSNet')
        net.renameVar('x0', 'input');
        net.renameVar('x12', 'prediction');
    else
        net.removeLayer(net.layers(end).name) ;
    end
    
    net.mode = 'train';
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
    
    for i = 1:1:length(filePaths)
        
        %%% read images
        image = imread(fullfile('testsets', folderTest, filePaths(i).name));
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        allName{count} = nameCur;
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
        net.conserveMemory = false; 
        tic
        net.eval({'input', input}) ;
        time(i) = toc;
        
        %% =========== Compressed Measurement ==========================
        Y       = squeeze(gather(net.vars(10).value)); 
        
        edges   = [-8:0.1:8];
        his     = histogram(Y(:), edges); 
        value   = his.Values./sum(his.Values); 
        range   = [min(Y(:)), max(Y(:))]; 
        if ~exist(['histResults\' network '\']), mkdir(['histResults\' network '\']); end
        save(['histResults\' network '\' nameCur '_r' num2str(subRate) '.mat'], 'his', 'value', 'range', 'Y');
        
%        %% =========== Decomposition Measurement ==========================
% 
%         im1     = squeeze(gather(net.vars(2).value));  %imagesc(im1); axis square;axis off; colormap(gray); colorbar;
%         %im      = export_fig; imwrite(im, ['Decomp\1' modelName '_rate' num2str(subRate) '_grey.tif']);
%         
%         im2     = squeeze(gather(net.vars(4).value));  %imagesc(im2); axis square;axis off; colormap(gray); colorbar;
%         %im      = export_fig; imwrite(im, ['Decomp\2' modelName '_rate' num2str(subRate) '_grey.tif']);
%         
%         im3     = squeeze(gather(net.vars(6).value));  %imagesc(im3); axis square;axis off; colormap(gray); colorbar;
%         %im      = export_fig; imwrite(im, ['Decomp\3' modelName '_rate' num2str(subRate) '_grey.tif']);
%         
%         im4     = squeeze(gather(net.vars(8).value));  %imagesc(im4); axis square;axis off; colormap(gray); colorbar;
%         %im      = export_fig; imwrite(im, ['Decomp\4' modelName '_rate' num2str(subRate) '_grey.tif']);
%         
%         im = cat(1, cat(2, im1, im2), cat(2, im3, im4)); imagesc(im); 
%         axis square;axis off; colormap(gray); colorbar; set(gca, 'fontsize', 20);        
%         im      = export_fig; imwrite(im, ['Decomp\' modelName '_rate' num2str(subRate) '_grey.tif']);
%         
%         
%         
% %         out1 = net.getVarIndex('prediction') ;
% %         output = gather(squeeze(gather(net.vars(out1).value)));
%         
%         
%         %% Check the sampling matrix
%         
% %         Phit = squeeze(gather(net.params(3).value));
%         
% %         Phi = zeros(size(Phit, 1), size(Phit, 2),4, size(Phit, 3));
% %         Phi(:, :, 1, :) = Phit; 
% %         Phi(:, :, 2, :) = (gather(net.params(3).value));
% %         Phi(:, :, 3, :) = (gather(net.params(7).value));
% %         Phi(:, :, 4, :) = (gather(net.params(11).value));
%         
%         tLegend = {'1', '2', '3', '4'};
%         
%         index_Phi     = [3, 7, 11, 15]; 
%         
%         bigMtx = [];
%         for idx1 = 1:1:2
%             tmpBig = [];
%             for idx2 = 1:1:2
%                 idx = (idx1-1) * 2 + idx2;
%                 
%                 %pr = figure(idx);
%                 Phi = squeeze(gather(net.params(index_Phi(idx)).value));
%                 
%                 mtx = [];
%                 [w, h, c, k] = size(Phi);
%                 k0 = floor(sqrt(c));
%                 
%                 for i0 = 1:1:k0
%                     tmp = [];
%                     for i1 = 1:1:k0
%                         i2 = (i0 - 1)*k0 + i1;
%                         %pr = figure(1);
%                         Phi_i = Phi(:, :, i2);
%                         
%                         
%                         tmp   = [tmp, Phi_i];
%                       
%                     end
%                     mtx = [mtx; tmp];
%                 end
%                 
% %                tmpBig = [tmpBig, mtx] ;
%                 imagesc(mtx); axis square;axis off; 
%                 colormap(gray ); colorbar; set(gcf,'color','w');set(gca, 'fontsize', 26); 
%                 im = export_fig; 
%                 imwrite(im, ['mtxResults_grey\' num2str(size(Phi,1)) '_meas_' modelName '_rate' num2str(subRate) '_grey.tif']);
%             end
%  %           bigMtx = [bigMtx; tmpBig];
%         end
%         
% %         pr = figure(1); imshow(bigMtx, []); set(gcf,'color','w'); set(gca, 'fontsize', 18); 
% % 
% %         im = export_fig; 
% %         imwrite(im, ['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '.tif']);
% %         saveas(pr, ['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '.png']);
% %         saveas(pr, ['mtxResults_grey\meas_' modelName '_rate'  num2str(subRate) '.fig']);        
% %         save(['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '.mat'], 'Phi');  
%         
%         if subRate == 0.1
%             imagesc(bigMtx,[-0.5 0.5]); axis square;axis off; 
%             colormap(gray ); colorbar; set(gcf,'color','w');set(gca, 'fontsize', 26); 
%         elseif subRate == 0.2
%             imagesc(bigMtx, [-0.5 0.5]); axis square;axis off; 
%             colormap(gray); colorbar; set(gcf,'color','w');set(gca, 'fontsize', 26); 
%         end
%         im = export_fig; 
%         imwrite(im, ['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '_grey.tif']);
%         %% Check CS measurement
%         %meas = net.getVarIndex('conv2') ;
%         output = gather(squeeze(gather(net.vars(1).value))) ;
%         
        
    end
end
end