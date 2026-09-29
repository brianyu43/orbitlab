# C4 표현을 설명하는 한 장

90도 회전 R은 네 번 적용하면 항등이다. 슬롯 순환 rho도 동일하게 rho⁴=I다. 원본은 각 슬롯 k를 E_k(x)=h(R^(-k)x)로 정의한다. 이미지를 r번 회전하면 E_k(R^r x)=h(R^(r-k)x)=E_(k-r)(x)이므로 E(R^r x)=rho_r E(x)다.

Decoder는 D(z)=(1/4)Σ_k R^k d(z_k)다. z의 슬롯을 r만큼 바꾸면 k-r=j로 치환할 수 있으므로 D(rho_r z)=(1/4)Σ_j R^(j+r)d(z_j)=R^r D(z)가 된다. 학습 여부와 무관한 구조적 관계다. 실제 부동소수점 구현의 차이는 수치 검사로 확인한다.

정규화 DFT는 zhat_m=(1/2)Σ_k z_k exp(-2πimk/4)다. 슬롯 r 이동은 m 성분을 exp(-2πimr/4)배 한다. m=0은 불변, m=2는 90도에서 부호 반전, m=1/3은 실수 입력에서 켤레 쌍이다. 실수 복원을 위해 m=1/3을 함께 제거하거나 배율을 바꾼다. 이 분해는 군 작용의 분해이며 색·모양·위치의 의미적 분리가 자동으로 증명되지 않는다.

불변 평균만 남기면 encode가 회전에서 같아진다. 정확한 등변 decoder에 이 코드를 주면 decode 역시 모든 C4 회전에 불변인 이미지가 되어, 한 방향 물체를 정확히 복원할 수 없다. B3의 복원 손실은 구현 오류 없이도 생길 수 있다.

일반 AE의 임의 좌표에는 슬롯 rho를 강제할 이유가 없다. train에서 학습한 A90를 사용하고 독립 장면에서 A90 z≈z_rot, A90⁴ z≈z, D(A90 z)≈R x를 검사한다. feature standardization 후의 선형 A는 원래 latent에서는 역표준화를 포함한 affine 작용으로 해석한다.

Flow에서는 vG(z,t,c)=(1/4)Σ_g rho_g^-1 v(rho_g z,t,c)를 쓴다. c=모양·색은 회전 불변이므로 velocity가 equivariant다. 같은 채널 정규화를 모든 group 슬롯에 사용하고 coupled noise도 rho로 바꾸면 Euler ODE 경로가 rho와 교환한다. 이 성질은 생성 표본의 모양·색 정확성이나 다양성을 보장하지 않는다.

검증 증거: `original_smoke_mps.json`, `research_unit_checks.json`. 범위는 이미지 중심에 대한 전체 장면의 C4 회전이다.
