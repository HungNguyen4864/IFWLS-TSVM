function [Predict_Y,train_time,output_struct] = IFW_LSTSVM(TestX,DataTrain,FunPara)

obs =DataTrain(:,end);
A=DataTrain(obs==1,1:end-1);
B= DataTrain(obs~=1,1:end-1);

c1 = FunPara.c1;
c2 = FunPara.c2;
epsilon=FunPara.c3;
K=FunPara.k;
kerfPara = FunPara.kerfPara;
m1=size(A,1);
m2=size(B,1);
e1=ones(m1,1);
e2=ones(m2,1);


tic
[S1,S2]=non_linear_score_values(DataTrain,FunPara.kerfPara.pars);
[ro1,ro2]=linear_W_interclass_weights(DataTrain,FunPara.kerfPara.pars,K);

if strcmp(kerfPara.type,'lin')
    P=diag(ro1)*[A,e1];
    Q=diag(S2)*[B,e2];
    H1=P'*P;
    Q1=Q'*Q;
    kerH1=-(H1+c1*Q1+epsilon*eye(length(Q1)))\(Q'*(diag(S2))*(e2));
    w1=kerH1(1:size(kerH1,1)-1,:);
    b1=kerH1(size(kerH1,1));
    P=diag(S1)*[A,e1];
    Q=diag(ro2)*[B,e2];
    H1=P'*P;
    Q1=Q'*Q;
    kerH2=(Q1+c2*H1+epsilon*eye(length(H1)))\(P'*(diag(S1))*(e1));
    w2=kerH2(1:size(kerH2,1)-1,:);
    b2=kerH2(size(kerH2,1));
else
    X=[A;B];
    KA=[kernelfun(A,kerfPara,X),e1];
    KB=[kernelfun(B,kerfPara,X),e2];
    
    P=diag(ro1)*KA;
    Q=diag(S2)*KB;
    H1=P'*P;
    Q1=Q'*Q;
    kerH1=-(H1+c1*Q1+epsilon*eye(length(Q1)))\(Q'*(diag(S2))*(e2));
    w1=kerH1(1:size(kerH1,1)-1,:);
    b1=kerH1(size(kerH1,1));
    P=diag(S1)*KA;
    Q=diag(ro2)*KB;
    H1=P'*P;
    Q1=Q'*Q;
    kerH2=(Q1+c2*H1+epsilon*eye(length(H1)))\(P'*(diag(S1))*(e1));
    w2=kerH2(1:size(kerH2,1)-1,:);
    b2=kerH2(size(kerH2,1));
end
train_time=toc;

if strcmp(kerfPara.type,'lin')
    P_1=TestX(:,1:end-1);
    w11=sqrt(w1'*w1);
    w22=sqrt(w2'*w2);
    y1=(P_1*w1+b1)/w11;
    y2=(P_1*w2+b2)/w22;
else
    C=[A;B];
    P_1=kernelfun(TestX(:,1:end-1),kerfPara,C);
    w11=sqrt(w1'*kernelfun(C,kerfPara,C)*w1);
    w22=sqrt(w2'*kernelfun(C,kerfPara,C)*w2);
    y1=(P_1*w1+b1)/w11;
    y2=(P_1*w2+b2)/w22;
end
Predict_Y = sign(abs(y2)-abs(y1));

st = dbstack;
output_struct.function_name= st.name;
end