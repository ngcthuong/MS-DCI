clear; close all; 
method = 'CSNet'; 
subrate = 0.2;

data = load(['mtxResults\meas_' method '_r' num2str(subrate) '_rate' num2str(subrate) '.mat']);
for i = 1:1:size(data.Phi, 3)
    tmp = data.Phi(:, :, i); 
    tmp_var(i) = var(tmp(:)); 
end

pr = figure(1); 
plot(sort(tmp_var(:)), 'k', 'LineWidth', 2); 
grid on; hold on; 

method = 'MS-CSNet3'; 
data = load(['mtxResults\meas_' method '_r' num2str(subrate) '_rate' num2str(subrate) '.mat']);
% first section 
for i = 1:1:size(data.Phi, 4)
    tmp = data.Phi(:, :, 1, i); 
    tmp_var1(i) = var(tmp(:)); 
    
    tmp = data.Phi(:, :, 2, i); 
    tmp_var2(i) = var(tmp(:)); 
    
    tmp = data.Phi(:, :, 3, i); 
    tmp_var3(i) = var(tmp(:)); 
    
    tmp = data.Phi(:, :, 4, i); 
    tmp_var4(i) = var(tmp(:)); 
end

% figure(2); 
plot(sort(tmp_var1(:)), 'r', 'LineWidth', 2); %hold on; 
plot(sort(tmp_var2(:)), 'b', 'LineWidth', 2);  
plot(sort(tmp_var3(:)), 'm', 'LineWidth', 2);  
plot(sort(tmp_var4(:)), 'g', 'LineWidth', 2); 
grid on; 
legend('CSNet', 'W-DCS-LL', 'W-DCS-LH', 'W-DCS-HL', 'W-DCS-HH'); set(gca,'fontsize', 14);
hold off; axis tight
saveas(pr, ['rate' num2str(subrate) '.png']);
saveas(pr, ['rate' num2str(subrate) '.fig']);