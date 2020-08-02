acc_body_x = importdata('UCI HAR Dataset/train/Inertial Signals/body_acc_x_train.txt',' ');
acc_body_y = importdata('UCI HAR Dataset/train/Inertial Signals/body_acc_y_train.txt',' ');
acc_body_z = importdata('UCI HAR Dataset/train/Inertial Signals/body_acc_z_train.txt',' ');
acc_total_x = importdata('UCI HAR Dataset/train/Inertial Signals/total_acc_x_train.txt',' ');
acc_total_y = importdata('UCI HAR Dataset/train/Inertial Signals/total_acc_y_train.txt',' ');
acc_total_z = importdata('UCI HAR Dataset/train/Inertial Signals/total_acc_z_train.txt',' ');
gyro_x = importdata('UCI HAR Dataset/train/Inertial Signals/body_gyro_x_train.txt',' ');
gyro_y = importdata('UCI HAR Dataset/train/Inertial Signals/body_gyro_y_train.txt',' ');
gyro_z = importdata('UCI HAR Dataset/train/Inertial Signals/body_gyro_z_train.txt',' ');
labels = importdata('UCI HAR Dataset/train/y_train.txt',' ');

activity_images = zeros(7352,36,128);
sequences = cat(3,acc_body_x, acc_body_y, acc_body_z, acc_total_x, acc_total_y, acc_total_z, gyro_x, gyro_y, gyro_z);
% sequences = sequences(:,1:2:end,:);


for x = 1:7352

    i = 1; j = i + 1; 
    SIS = int2str(i);
    SI = sequences (x,:,i);
    indx = i;
    ns = 9;
    
    while i ~= j
        if j > ns
            j = 1;
        else
            if isempty(strfind(SIS,strcat(int2str(i),int2str(j)))) && isempty(strfind(SIS,strcat(int2str(j),int2str(i))))
                SI = [SI;sequences(x,:,j)];
                SIS = strcat(SIS,int2str(j));
                i=j;j=i+1;
            else 
             j = j + 1;    
            end
        end
    end
    SI(37,:) = [];
    activity_images(x,:,:) = log(abs(fftshift(fft2(SI))));
end

for i =1: 7352
    temp = squeeze(activity_images(i,:,:));
    save(strcat('original .mat/',int2str(i),'.mat'),'temp');
end

     
