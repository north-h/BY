#include<bits/stdc++.h>

using namespace std;

int main(){
	char a;
	long long s=1;
	cin>>a;
	long long n=a-64;
	for(int i=1;i<=n;i++){
		for(int j=1;j<=n-i;j++){
			cout<<' ';
		}
		if(i==n){
			for(int j=1;j<=n*2-1;j++){
				char m='A'+i-1;
				cout<<m;
			}
			return 0;
		}
		for(int j=1;j<=2;j++){
			if(i==1){
				cout<<"A";
				break;
			}
			if(j==2){
				for(int k=1;k<=s-2;k++){
					cout<<' ';
				}
			}
			char m='A'+i-1;
			cout<<m;
		}
		cout<<'\n';
		s+=2;
	}
	return 0;
}
