#include<bits/stdc++.h>
using namespace std;
int a,b,d;
char c;
int main(){
	cin>>a>>b>>c>>d;
	if(d==1){
		for(int i=1;i<=a;i++){
			for(int j=1;j<=b;j++){
				cout<<c;
			}
			cout<<endl; 
		}
	}else{
		for(int i=1;i<=a;i++){
			for(int j=1;j<=b;j++){
				if(i==1||i==a||j==1||j==b){
					cout<<c;
				}else{
					cout<<" ";
				}
			}
			cout<<endl; 
		}
	}
}
