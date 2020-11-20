% demo show measurement
clear
for netId = [3]
network = ['SS-DCS' num2str(netId)];
% network = 'CSNet'; 
for subRate = [ 0.2 0.3]
    modelName   = [network '_r' num2str(subRate)];
    
    folderTest  = 'Classic13_512';
    ext         =  {'*.jpg','*.png','*.bmp', '*.pgm', '*.tif'};
    filePaths   =  [];
    for i = 1 : length(ext)
        filePaths = cat(1,filePaths, dir(fullfile('testsets',folderTest,ext{i})) );
    end
    
    for i = 1:1:4%numel(filePaths)
        [~,nameCur,extCur] = fileparts(filePaths(i).name);
        data = load(['histResults\' network '\' nameCur '_r' num2str(subRate) '.mat' ]);
        
        noDim = floor(sqrt(size(data.Y, 3)));
        
        [m, n, c] = size(data.Y); 
        data.Y   = data.Y(randperm(m), randperm(n), randperm(c)); 
        
        t_im1 = [];
        for i1 = 1:1:size(data.Y, 1)
            t_im2 = [];
            for i2 = 1:1:size(data.Y, 2)
                im_ = reshape(squeeze(data.Y(i1, i2, 1:noDim^2)), [noDim, noDim]);
                t_im2 = [t_im2, im_];
            end
            t_im1 = [t_im1; t_im2];
        end
        
        imagesc(t_im1); axis square;axis off; colormap(gray); colorbar; set(gca, 'fontsize', 20);set(gcf,'color','w');

        if ~exist(['Results_Meas_Analysis\' network '\']), mkdir(['Results_Meas_Analysis\' network '\']); end
        im      = export_fig; imwrite(im, ['Results_Meas_Analysis\' network '\' nameCur '_r' num2str(subRate) '_grey.tif']);
    end
end
end