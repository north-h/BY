#include <bits/stdc++.h>
using namespace std;
int main(){
	long long n,a;
	cin>>n;
	a=n;
	for(int c=1;c<=n;c++){
		for(int q=1;q<=n;q++){
			if(c==1||c==n||q==a){
				cout<<"*";
			}
			else{
				cout<<" ";
			}
		}
		cout<<endl;
		a--;
	}
	return 0;
}
