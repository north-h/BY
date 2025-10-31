#include<bits/stdc++.h>

using namespace std;

int a[1000010], b[1000010];

bool cmp(int a, int b) {
	if (abs(a) != abs(b)) return abs(a) < abs(b);
	else return a < b;
}

int main() {
	int n; cin >> n;
	for (int i = 0; i < n; i ++) cin >> a[i];
	sort(a, a + n, cmp);
	for (int i = 0; i < n; i ++) cout << a[i] << ' ';
	return 0;
}


