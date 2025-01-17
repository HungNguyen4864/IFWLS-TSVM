function [ro1,ro2]=linear_W_interclass_weights(A,mew,k)
A1=A(A(:,end)==1,:);
B1=A(A(:,end)~=1,:);

D = pdist(A1(:,1:end-1));
Z = squareform(D);

[Asorted, OrigColIdx] = sort(Z,2);
aak=Asorted(:,2:2+k);
exaak=exp(-aak.^2/mew);
ro1=sum(exaak,2);

D = pdist(B1(:,1:end-1));
Z = squareform(D);

[Asorted, OrigColIdx] = sort(Z,2);
aak=Asorted(:,2:2+k);
exaak=exp(-aak.^2/mew);
ro2=sum(exaak,2);
end