
n = 8;
alist = [0.25 0.5 1]; % values of alpha to plot
blist = [ 0:0.05:1 ]; % values of beta to plot

figure(1)
clf

for ai = 1:length(alist)
    subplot(2,3,ai)
    bmat = zeros(n+1,length(blist)); % matrix of stationary distributions as a function of beta
    for bi = 1:length(blist)
    
a = alist(ai);
b = blist(bi);

T = zeros(n+1,n+1);

for i = 0:n
    for j = 0:n
        T(i+1,j+1) = binopdf(j,n,1/(1+exp(-a*(i-n/2)-b))); % transition matrix of Markov chain
    end
end

[v d] = eig(T'); % eigenvectors of transition matrix
d = diag(d);
[val ind] = sort(d);
statdist = v(:,ind(end));  % eigenvector with highest eigenvalue
statdist = statdist/sum(statdist); % normalized to stationary distribution
bmat(:,bi) = statdist;
    end
    colormap hot
   pcolor([ bmat zeros(n+1,1); zeros(1,length(blist)+1)]);
   shading flat
   subinds=1:2:length(blist)
   set(gca,'ytick',[0:1:n]+1.5,'yticklabels',[0:1:n],'xtick',subinds+0.5,'xticklabels',blist(subinds))
   xlabel('{\it \beta }')
   ylabel('{\it k}')
   title([ '{\it \alpha} = ' num2str(alist(ai))])
   colorbar
   subplot(2,3,3+ai)
   plot(blist,1./(1+exp(-blist)),'k-',blist,(0:n)*bmat/n,'k:')
   set(gca,'ylim',[0.5 1],'xtick',blist(subinds),'xticklabels',blist(subinds))
    xlabel('{\it \beta }')
   ylabel('Proportion')
   title([ '{\it \alpha} = ' num2str(alist(ai))])
end
legend('Individual bias','Mean group bias')

