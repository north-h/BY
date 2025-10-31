#include<bits/stdc++.h>

using namespace std;

int main() {
	int n;
	cin >> n;
	char op = 'a';
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= 26; j ++) {
			cout << op;
			op ++;
		}
		cout << '\n';
		op --;
		for (int j = 1; j <= 26; j ++) {
			cout << op;
			op --;
		}
		cout << '\n';
		op ++;
	}
	return 0;
}


