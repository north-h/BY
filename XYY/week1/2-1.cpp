#include<bits/stdc++.h>
using namespace std;
string s[100005];
int main() {
	int n;
	cin >> n;
	for (int i = 1; i <= n; i ++) {
		cin >> s[i];
	}
	for (int i = 1; i <= n; i ++) {
		cout << s[i] << endl;
	}
	int a, b;
	while (cin >> a >> b) {
		n ++;
		s[n] = s[a] + s[b];
		cout << s[n] << endl;
	}
	return 0;
}
