#include<bits/stdc++.h>
using namespace std;
int main() {
	int n; cin >> n;
	int cnt1 = n, cnt2 = 0;
	for (int i = 1; i <= n; i ++) {
		for (int j = 1; j <= cnt2; j ++) cout << ' ';
		for (int j = 1; j <= cnt1; j ++) cout << '*';
		cout << '\n';
		cnt1 --, cnt2 ++;
	}
	return 0;
}
