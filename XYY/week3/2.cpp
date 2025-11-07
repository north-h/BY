#include<bits/stdc++.h>

using namespace std;

int main() {
	int n; cin >> n;
	int b = -100010, sum = 0, ans = 0;
	for (int i  = 1; i <= n; i ++) {
		int a; cin >> a;
		if (a > b) {
			sum ++;
		} else {
			sum = 1;
		}
		ans = max(ans, sum);
		b = a; 
	}
	cout << ans << '\n';
	return 0;
}
