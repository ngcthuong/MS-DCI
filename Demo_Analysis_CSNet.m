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
networkTest = {'CSNet'};      % 10

showResult  = 0;
writeRecon  = 1;
featureSize = 64;
blkSize     = 32;
isLearnMtx  = [1, 0];
network     = networkTest{1};

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
        out1 = net.getVarIndex('prediction') ;
        output = gather(squeeze(gather(net.vars(out1).value)));
        
        
         %% =========== Compressed Measurement ==========================
        Y       = squeeze(gather(net.vars(2).value)); 
        
        edges   = [-8:0.1:8];
        his     = histogram(Y(:), edges); 
        value   = his.Values./sum(his.Values); 
        range   = [min(Y(:)), max(Y(:))]; 
        if ~exist(['histResults\' network '\']), mkdir(['histResults\' network '\']); end
        save(['histResults\' network '\' nameCur '_r' num2str(subRate) '.mat'], 'his', 'value', 'range', 'Y');
        
%         %% =========== Decomposition ===================================
        
        %% Check the sampling matrix
%         Phi = squeeze(gather(net.params(1).value));
%         tLegend = {'Low-Low', 'Low-High', 'High-Low', 'High-High'};
%         
%         [w, h, c, k] = size(Phi);
%         k0 = floor(sqrt(c));
%         
%         bigMtx = [];
%         for idx1 = 1:1:1
%             tmpBig = [];
%             for idx2 = 1:1:1
%                 idx = (idx1-1) * 2 + idx2;
%                 
%                 pr = figure(idx);
%                 mtx = [];
%                 for i0 = 1:1:k0
%                     tmp = [];
%                     for i1 = 1:1:k0
%                         i2 = (i0 - 1)*k0 + i1;
%                         
%                         Phi_i = Phi(:, :, i2);                        
%                         %% padding zero to image 
%                         %Phi_i = padarray(Phi_i, [0, 4], 0, 'post');                         
%                         tmp   = [tmp, Phi_i];
%                        
%                     end
%                     %tmp = padarray(tmp, [4, 0], 0, 'post'); 
%                     
%                     mtx = [mtx; tmp];
%                 end
%                 tmpBig = [tmpBig, mtx] ;
%                 imshow(mtx, []); title([tLegend{idx}]);
%             end
%             bigMtx = [bigMtx; tmpBig];
%         end
%         
%         pr = figure(1); imshow(bigMtx, []);
%         saveas(pr, ['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '.png']);
%         saveas(pr, ['mtxResults_grey\meas_' modelName '_rate'  num2str(subRate) '.fig']);
%         im = export_fig; %color bar; 
%         imwrite(im, ['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '.tif']); 
%         save(['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '.mat'], 'Phi');
%         
%         if subRate == 0.1
%             imagesc(bigMtx, [-.2 .2]); axis square;axis off; 
%         elseif subRate == -0.2
%             imagesc(bigMtx, [-0.25 0.25]); axis square;axis off; 
%         end
%         colormap(gray); colorbar; set(gcf,'color','w');set(gca, 'fontsize', 26); 
%         im = export_fig; 
%         imwrite(im, ['mtxResults_grey\meas_' modelName '_rate' num2str(subRate) '_grey.tif']);
        
    end
end
