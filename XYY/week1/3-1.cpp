#include<bits/stdc++.h>
using namespace std;
int main(){
	int n;
	cin>>n;
	for(int i=1;i<=n;i++){
		char ai1='a',ai2='z';
		for(int j=1;j<=26;j++){
			cout<<ai1;
			ai1++;
		}
		cout<<endl;
		for(int j=1;j<=26;j++){
			cout<<ai2;
			ai2--;
		}
		cout<<endl;
	}
	return 0;
}
