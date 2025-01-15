function [accuracy,alpha_d]=nonL_WIFLSTWSVM(A,A_test,FunPara)
obs =A(:,end);
A1=A(obs==1,1:end-1);
B1= A(obs~=1,1:end-1);

C1 = FunPara.c1;
C2 = FunPara.c3;
K=FunPara.k;
mew = FunPara.kerfPara.pars;

C3=C1;C4=C2;

x0=A(:,1:end-1);y0=A(:,end);
xtest0=A_test(:,1:end-1);ytest0=A_test(:,end);
Cf=[x0 y0];
tic
[S1,S2,alpha_d]=linear_score_values(A,mew);
[ro1,ro2]=linear_W_interclass_weights(A,mew,K);
time1=toc;

C=[A1;B1];
[x y]=size(A1);
[x1 y1]=size(B1);
X=[A1;B1];
K1=kernelfun(A1,kerfPara,X);
K2=kernelfun(B1,kerfPara,X);

K3 = exp(-(1/(mew^2))*(repmat(sqrt(sum(A_test(:,1:end-1).^2,2).^2),1,size(A,1))-2*(A_test(:,1:end-1)*A(:,1:end-1)')+repmat(sqrt(sum(A(:,1:end-1).^2,2)'.^2),size(A_test,1),1)));
K4 = exp(-(1/(mew^2))*(repmat(sqrt(sum(A(:,1:end-1).^2,2).^2),1,size(A,1))-2*(A(:,1:end-1)*A(:,1:end-1)')+repmat(sqrt(sum(A(:,1:end-1).^2,2)'.^2),size(A,1),1)));

m1=size(A1,1);m2=size(B1,1);m3=size(C,1);
e1=ones(m1,1);e2=ones(m2,1);

T=[S2.*K2 S2.*e2];
TtT=T'*T;
R=[ro1.*K1 ro1.*e1];
RtR=R'*R;
I=eye(size(RtR,1));
u1=-(C1.*TtT+RtR+C2.*I)\T'*(S2.*e2);
T=[ro2.*K2 ro2.*e2];
TtT=T'*T;
R=[S1.*K1 S1.*e1];RtR=R'*R;
I=eye(size(RtR,1));
u2=(C3.*TtT+RtR+C4.*I)\R'*(S1.*e1);
train_Time=time1+toc;


no_test=size(xtest0,1);
preY1=[];
preY2=[];

preY2=(K3*u2(1:size(u2,1)-1,:)+u2(size(u2,1),:))/sqrt(u2(1:size(u2,1)-1,:)'*K4*u2(1:size(u2,1)-1,:));
preY1=(K3*u1(1:size(u1,1)-1,:)+u1(size(u1,1),:))/sqrt(u1(1:size(u1,1)-1,:)'*K4*u1(1:size(u1,1)-1,:));
predicted_class=[];
for i=1:no_test
    if abs(preY1(i))< abs(preY2(i))
        predicted_class=[predicted_class;1];
    else
        predicted_class=[predicted_class;-1];
    end
    
end
err = sum(predicted_class ~= ytest0);
accuracy=(no_test-err)/(no_test)*100

return
end
