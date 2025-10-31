#include <bits/stdc++.h>
#define int long long

void solve () {
	
	int n;
	std::cin >> n;
	
	for(int i = 1; i <= n; i++) {
		for(int j = 1; j <= i; j++) {
			if(j == 1 || j == i) {
				std::cout << "*";
			}else {
				std::cout << " ";
			}
		}
		for(int j = 1; j < n - i; j++) {
			std::cout << " ";
		}
		if(i != n) std::cout << "*";
		std::cout << '\n';
	}
}

signed main() {
	std::ios::sync_with_stdio(false);
	std::cin.tie(nullptr);
	
	int t = 1;
	// std::cin >> t;
	for (int i = 0; i < t; i++) {
		solve ();
	}
	
	return 0;
}
